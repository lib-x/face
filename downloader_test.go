package face

import (
	"crypto/md5"
	"encoding/hex"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Test: ModelDownloader initialization

func TestNewModelDownloader(t *testing.T) {
	outputDir := "./testdata"
	defer os.RemoveAll(outputDir)

	downloader := NewModelDownloader(outputDir)

	if downloader.OutputDir != outputDir {
		t.Errorf("Expected output dir %s, got %s", outputDir, downloader.OutputDir)
	}

	if downloader.Timeout != 10*time.Minute {
		t.Errorf("Expected default timeout 10m, got %v", downloader.Timeout)
	}

	if downloader.SkipVerification {
		t.Error("Expected SkipVerification to be false by default")
	}
}

// Test: Available models

func TestAvailableModels(t *testing.T) {
	requiredModels := []string{"pigo-facefinder", "openface"}

	for _, key := range requiredModels {
		model, exists := AvailableModels[key]
		if !exists {
			t.Errorf("Required model '%s' not found in AvailableModels", key)
			continue
		}

		// Verify model info is complete
		if model.Name == "" {
			t.Errorf("Model '%s' has empty name", key)
		}

		if model.URL == "" {
			t.Errorf("Model '%s' has empty URL", key)
		}

		if model.Filename == "" {
			t.Errorf("Model '%s' has empty filename", key)
		}

		if !strings.HasPrefix(model.URL, "http://") && !strings.HasPrefix(model.URL, "https://") {
			t.Errorf("Model '%s' has invalid URL: %s", key, model.URL)
		}
	}
}

func TestModelInfo_Structure(t *testing.T) {
	tests := []struct {
		key          string
		expectedType ModelType
		minSize      int64
		requiresMD5  bool
	}{
		{"pigo-facefinder", "", 50000, false},       // ~50KB
		{"openface", ModelOpenFace, 30000000, true}, // ~30MB
	}

	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			model, exists := AvailableModels[tt.key]
			if !exists {
				t.Fatalf("Model '%s' not found", tt.key)
			}

			if tt.expectedType != "" && model.ModelType != tt.expectedType {
				t.Errorf("Expected model type %s, got %s", tt.expectedType, model.ModelType)
			}

			if model.Size < tt.minSize {
				t.Errorf("Expected size >= %d, got %d", tt.minSize, model.Size)
			}

			if tt.requiresMD5 && model.MD5 == "" {
				t.Errorf("Model '%s' should have MD5 checksum", tt.key)
			}
		})
	}
}

// Test: Download with mock server

func TestDownloadModel_MockServer(t *testing.T) {
	// Create test data
	testData := []byte("test model file content")
	testMD5 := calculateMD5(testData)

	// Create mock HTTP server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", string(rune(len(testData))))
		w.Write(testData)
	}))
	defer server.Close()

	// Create temp output directory
	outputDir, err := ioutil.TempDir("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(outputDir)

	// Create downloader
	downloader := NewModelDownloader(outputDir)
	downloader.Timeout = 5 * time.Second

	// Create mock model info
	testModel := ModelInfo{
		Name:     "Test Model",
		URL:      server.URL,
		Filename: "test_model.dat",
		MD5:      testMD5,
		Size:     int64(len(testData)),
	}

	// Test download
	err = downloader.DownloadModel(testModel)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Verify file exists
	outputPath := filepath.Join(outputDir, testModel.Filename)
	if !fileExists(outputPath) {
		t.Fatal("Downloaded file does not exist")
	}

	// Verify file content
	downloadedData, err := ioutil.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("Failed to read downloaded file: %v", err)
	}

	if string(downloadedData) != string(testData) {
		t.Error("Downloaded content does not match original")
	}
}

func TestDownloadModel_ProgressCallback(t *testing.T) {
	testData := make([]byte, 1024*100) // 100KB
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", string(rune(len(testData))))
		// Write in chunks to test progress
		chunkSize := 1024 * 10
		for i := 0; i < len(testData); i += chunkSize {
			end := i + chunkSize
			if end > len(testData) {
				end = len(testData)
			}
			w.Write(testData[i:end])
			time.Sleep(10 * time.Millisecond) // Simulate slow download
		}
	}))
	defer server.Close()

	outputDir, err := ioutil.TempDir("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(outputDir)

	downloader := NewModelDownloader(outputDir)
	downloader.SkipVerification = true

	progressCalled := false
	progressCount := 0
	downloader.OnProgress = func(progress DownloadProgress) {
		progressCalled = true
		progressCount++

		// Verify progress structure
		if progress.Downloaded < 0 || progress.Downloaded > progress.Total {
			t.Errorf("Invalid downloaded bytes: %d (total: %d)",
				progress.Downloaded, progress.Total)
		}

		if progress.Total > 0 && (progress.Percentage < 0 || progress.Percentage > 100) {
			t.Errorf("Invalid percentage: %.2f", progress.Percentage)
		}
	}

	testModel := ModelInfo{
		Name:     "Test Model",
		URL:      server.URL,
		Filename: "test_progress.dat",
		Size:     int64(len(testData)),
	}

	err = downloader.DownloadModel(testModel)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	if !progressCalled {
		t.Error("Progress callback was not called")
	}

	if progressCount < 2 {
		t.Errorf("Expected multiple progress updates, got %d", progressCount)
	}
}

func TestDownloadModel_ExistingFile(t *testing.T) {
	outputDir, err := os.MkdirTemp("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(outputDir)

	// Create existing file
	filename := "existing_model.dat"
	existingPath := filepath.Join(outputDir, filename)
	testData := []byte("existing content")
	if err := ioutil.WriteFile(existingPath, testData, 0644); err != nil {
		t.Fatalf("Failed to create existing file: %v", err)
	}

	downloader := NewModelDownloader(outputDir)
	downloader.SkipVerification = true

	testModel := ModelInfo{
		Name:     "Test Model",
		URL:      "http://example.com/model",
		Filename: filename,
	}

	// Should not re-download
	err = downloader.DownloadModel(testModel)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Verify content unchanged
	content, _ := ioutil.ReadFile(existingPath)
	if string(content) != string(testData) {
		t.Error("Existing file was modified")
	}
}

func TestDownloadModel_MD5Verification(t *testing.T) {
	testData := []byte("test content for MD5")
	correctMD5 := calculateMD5(testData)
	incorrectMD5 := "incorrect_md5_checksum"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(testData)
	}))
	defer server.Close()

	t.Run("Correct MD5", func(t *testing.T) {
		outputDir, err := ioutil.TempDir("", "model_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(outputDir)

		downloader := NewModelDownloader(outputDir)
		testModel := ModelInfo{
			Name:     "Test Model",
			URL:      server.URL,
			Filename: "test_correct_md5.dat",
			MD5:      correctMD5,
		}

		err = downloader.DownloadModel(testModel)
		if err != nil {
			t.Errorf("Download should succeed with correct MD5: %v", err)
		}
	})

	t.Run("Incorrect MD5", func(t *testing.T) {
		outputDir, err := ioutil.TempDir("", "model_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(outputDir)

		downloader := NewModelDownloader(outputDir)
		testModel := ModelInfo{
			Name:     "Test Model",
			URL:      server.URL,
			Filename: "test_incorrect_md5.dat",
			MD5:      incorrectMD5,
		}

		err = downloader.DownloadModel(testModel)
		if err == nil {
			t.Error("Download should fail with incorrect MD5")
		}

		if !strings.Contains(err.Error(), "checksum") {
			t.Errorf("Error should mention checksum, got: %v", err)
		}
	})

	t.Run("Skip Verification", func(t *testing.T) {
		outputDir, err := ioutil.TempDir("", "model_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(outputDir)

		downloader := NewModelDownloader(outputDir)
		downloader.SkipVerification = true

		testModel := ModelInfo{
			Name:     "Test Model",
			URL:      server.URL,
			Filename: "test_skip_verify.dat",
			MD5:      incorrectMD5,
		}

		err = downloader.DownloadModel(testModel)
		if err != nil {
			t.Errorf("Download should succeed when verification is skipped: %v", err)
		}
	})
}

func TestDownload_ByKey(t *testing.T) {
	outputDir, err := ioutil.TempDir("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(outputDir)

	downloader := NewModelDownloader(outputDir)

	// Test invalid key
	err = downloader.Download("non_existent_model")
	if err == nil {
		t.Error("Expected error for non-existent model key")
	}
}

func TestGetModelPath(t *testing.T) {
	outputDir := "/path/to/models"

	tests := []struct {
		modelKey     string
		expectedPath string
		expectError  bool
	}{
		{
			"pigo-facefinder",
			filepath.Join(outputDir, "facefinder"),
			false,
		},
		{
			"openface",
			filepath.Join(outputDir, "nn4.small2.v1.t7"),
			false,
		},
		{
			"non_existent",
			"",
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.modelKey, func(t *testing.T) {
			path, err := GetModelPath(outputDir, tt.modelKey)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got nil")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if path != tt.expectedPath {
					t.Errorf("Expected path %s, got %s", tt.expectedPath, path)
				}
			}
		})
	}
}

// Test: Utility functions

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		bytes    int64
		expected string
	}{
		{0, "0 B"},
		{500, "500 B"},
		{1024, "1.0 KB"},
		{1536, "1.5 KB"},
		{1048576, "1.0 MB"},
		{1572864, "1.5 MB"},
		{1073741824, "1.0 GB"},
		{1099511627776, "1.0 TB"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := formatBytes(tt.bytes)
			if result != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestFormatSpeed(t *testing.T) {
	tests := []struct {
		bytesPerSec float64
		contains    string
	}{
		{100, "100 B/s"},
		{1024, "1.0 KB/s"},
		{1048576, "1.0 MB/s"},
	}

	for _, tt := range tests {
		t.Run(tt.contains, func(t *testing.T) {
			result := formatSpeed(tt.bytesPerSec)
			if !strings.Contains(result, tt.contains) {
				t.Errorf("Expected result to contain %s, got %s", tt.contains, result)
			}
		})
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		duration time.Duration
		expected string
	}{
		{30 * time.Second, "30s"},
		{90 * time.Second, "1m 30s"},
		{125 * time.Second, "2m 5s"},
		{3600 * time.Second, "60m 0s"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := formatDuration(tt.duration)
			if result != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, result)
			}
		})
	}
}

// Test: Error handling

func TestDownloadModel_NetworkError(t *testing.T) {
	outputDir, err := ioutil.TempDir("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(outputDir)

	downloader := NewModelDownloader(outputDir)
	downloader.Timeout = 1 * time.Second

	testModel := ModelInfo{
		Name:     "Test Model",
		URL:      "http://invalid-url-that-does-not-exist.com/model",
		Filename: "test_error.dat",
	}

	err = downloader.DownloadModel(testModel)
	if err == nil {
		t.Error("Expected error for invalid URL")
	}
}

func TestDownloadModel_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	outputDir, err := ioutil.TempDir("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(outputDir)

	downloader := NewModelDownloader(outputDir)

	testModel := ModelInfo{
		Name:     "Test Model",
		URL:      server.URL,
		Filename: "test_404.dat",
	}

	err = downloader.DownloadModel(testModel)
	if err == nil {
		t.Error("Expected error for 404 response")
	}
}

// Benchmark tests

func BenchmarkFormatBytes(b *testing.B) {
	testBytes := int64(1572864) // 1.5 MB

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		formatBytes(testBytes)
	}
}

func BenchmarkCalculateMD5(b *testing.B) {
	testData := make([]byte, 1024*1024) // 1MB
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculateMD5(testData)
	}
}

// Helper functions

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func calculateMD5(data []byte) string {
	hash := md5.Sum(data)
	return hex.EncodeToString(hash[:])
}

// Integration test (requires network, run with -integration flag)

func TestDownloadRequired_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	outputDir, err := ioutil.TempDir("", "model_integration")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(outputDir)

	downloader := NewModelDownloader(outputDir)
	downloader.Timeout = 2 * time.Minute // Longer timeout for real downloads

	t.Log("Starting download of required models (this may take a while)...")

	err = downloader.DownloadRequired()
	if err != nil {
		t.Fatalf("Failed to download required models: %v", err)
	}

	// Verify files exist
	expectedFiles := []string{"facefinder", "nn4.small2.v1.t7"}
	for _, filename := range expectedFiles {
		path := filepath.Join(outputDir, filename)
		if !fileExists(path) {
			t.Errorf("Expected file not found: %s", path)
		}
	}
}
