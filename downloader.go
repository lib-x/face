package face

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"golang.org/x/net/proxy"
)

// ModelInfo contains information about a downloadable model
type ModelInfo struct {
	Name        string
	URL         string
	Filename    string
	MD5         string // Optional checksum
	Size        int64  // Expected size in bytes
	Description string
	ModelType   ModelType
}

// AvailableModels Available models for download
var AvailableModels = map[string]ModelInfo{
	"pigo-facefinder": {
		Name:        "Pigo Face Detector",
		URL:         "https://raw.githubusercontent.com/esimov/pigo/master/cascade/facefinder",
		Filename:    "facefinder",
		Size:        51764, // ~50KB
		Description: "Pigo cascade classifier for face detection",
	},
	"openface": {
		Name:        "OpenFace nn4.small2.v1",
		URL:         "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7",
		Filename:    "nn4.small2.v1.t7",
		MD5:         "c95bfd8cc1adf05210e979ff623013b6",
		Size:        31510785, // ~30MB
		Description: "OpenFace face recognition model (96x96, 128-dim)",
		ModelType:   ModelOpenFace,
	},
	"openface-alternative": {
		Name:        "OpenFace nn4.small2.v1 (Mirror)",
		URL:         "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7",
		Filename:    "nn4.small2.v1.t7",
		Size:        31510785,
		Description: "OpenFace model from alternative mirror",
		ModelType:   ModelOpenFace,
	},
	"openface-kde": {
		Name:        "OpenFace nn4.small2.v1 (KDE Mirror)",
		URL:         "https://files.kde.org/digikam/facesengine/dnnface/openface_nn4.small2.v1.t7",
		Filename:    "nn4.small2.v1.t7",
		Size:        31510785,
		Description: "OpenFace model from KDE mirror",
		ModelType:   ModelOpenFace,
	},
}

// DownloadProgress represents download progress
type DownloadProgress struct {
	Total      int64
	Downloaded int64
	Percentage float64
	Speed      float64 // bytes per second
	Elapsed    time.Duration
}

// ProgressCallback is called during download to report progress
type ProgressCallback func(progress DownloadProgress)

// ModelDownloader handles model file downloads
type ModelDownloader struct {
	OutputDir        string
	OnProgress       ProgressCallback
	Timeout          time.Duration
	SkipVerification bool
	ProxyURL         string // SOCKS5 or HTTP proxy URL (e.g., "socks5://127.0.0.1:10808")
}

// NewModelDownloader creates a new model downloader
func NewModelDownloader(outputDir string) *ModelDownloader {
	return &ModelDownloader{
		OutputDir:        outputDir,
		Timeout:          10 * time.Minute,
		SkipVerification: false,
	}
}

// Download downloads a model by its key
func (md *ModelDownloader) Download(modelKey string) error {
	model, exists := AvailableModels[modelKey]
	if !exists {
		return fmt.Errorf("model '%s' not found in available models", modelKey)
	}

	return md.DownloadModel(model)
}

// DownloadModel downloads a specific model
func (md *ModelDownloader) DownloadModel(model ModelInfo) error {
	// Create output directory
	if err := os.MkdirAll(md.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}

	outputPath := filepath.Join(md.OutputDir, model.Filename)

	// Check if file already exists
	if md.fileExists(outputPath) {
		fmt.Printf("File already exists: %s\n", outputPath)

		if !md.SkipVerification && model.MD5 != "" {
			fmt.Println("Verifying existing file...")
			if md.verifyMD5(outputPath, model.MD5) {
				fmt.Println("✓ File verification passed")
				return nil
			}
			fmt.Println("✗ File verification failed, re-downloading...")
			os.Remove(outputPath)
		} else {
			return nil
		}
	}

	fmt.Printf("Downloading %s...\n", model.Name)
	fmt.Printf("URL: %s\n", model.URL)
	fmt.Printf("Output: %s\n", outputPath)

	// Create HTTP client with timeout and proxy support
	client, err := md.createHTTPClient()
	if err != nil {
		return fmt.Errorf("failed to create HTTP client: %v", err)
	}

	// Make request
	resp, err := client.Get(model.URL)
	if err != nil {
		return fmt.Errorf("failed to download: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status: %s", resp.Status)
	}

	// Create output file
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer outFile.Close()

	// Download with progress tracking
	if err := md.downloadWithProgress(outFile, resp.Body, resp.ContentLength); err != nil {
		os.Remove(outputPath)
		return fmt.Errorf("download failed: %v", err)
	}

	fmt.Println("\n✓ Download completed")

	// Verify MD5 checksum if provided
	if !md.SkipVerification && model.MD5 != "" {
		fmt.Println("Verifying checksum...")
		if !md.verifyMD5(outputPath, model.MD5) {
			os.Remove(outputPath)
			return fmt.Errorf("checksum verification failed")
		}
		fmt.Println("✓ Checksum verified")
	}

	return nil
}

// downloadWithProgress downloads content with progress reporting
func (md *ModelDownloader) downloadWithProgress(dst io.Writer, src io.Reader, totalSize int64) error {
	startTime := time.Now()
	var downloaded int64

	buffer := make([]byte, 32*1024) // 32KB buffer
	lastUpdate := time.Now()

	for {
		n, err := src.Read(buffer)
		if n > 0 {
			if _, writeErr := dst.Write(buffer[:n]); writeErr != nil {
				return writeErr
			}
			downloaded += int64(n)

			// Update progress every 100ms
			if time.Since(lastUpdate) > 100*time.Millisecond {
				if md.OnProgress != nil {
					elapsed := time.Since(startTime)
					speed := float64(downloaded) / elapsed.Seconds()
					percentage := 0.0
					if totalSize > 0 {
						percentage = float64(downloaded) / float64(totalSize) * 100
					}

					md.OnProgress(DownloadProgress{
						Total:      totalSize,
						Downloaded: downloaded,
						Percentage: percentage,
						Speed:      speed,
						Elapsed:    elapsed,
					})
				} else {
					// Default progress output
					md.printProgress(downloaded, totalSize)
				}
				lastUpdate = time.Now()
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	return nil
}

// printProgress prints download progress
func (md *ModelDownloader) printProgress(downloaded, total int64) {
	if total > 0 {
		percentage := float64(downloaded) / float64(total) * 100
		fmt.Printf("\rProgress: %.1f%% (%s / %s)",
			percentage,
			formatBytes(downloaded),
			formatBytes(total))
	} else {
		fmt.Printf("\rDownloaded: %s", formatBytes(downloaded))
	}
}

// fileExists checks if a file exists
func (md *ModelDownloader) fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// verifyMD5 verifies the MD5 checksum of a file
func (md *ModelDownloader) verifyMD5(path, expectedMD5 string) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return false
	}

	actualMD5 := hex.EncodeToString(hash.Sum(nil))
	return actualMD5 == expectedMD5
}

// DownloadAll downloads all available models
func (md *ModelDownloader) DownloadAll() error {
	fmt.Printf("Downloading %d models...\n\n", len(AvailableModels))

	failed := make([]string, 0)

	for key, model := range AvailableModels {
		fmt.Printf("\n[%s]\n", key)
		if err := md.DownloadModel(model); err != nil {
			fmt.Printf("✗ Failed: %v\n", err)
			failed = append(failed, key)
			continue
		}
	}

	if len(failed) > 0 {
		return fmt.Errorf("failed to download %d model(s): %v", len(failed), failed)
	}

	fmt.Println("\n✓ All models downloaded successfully")
	return nil
}

// DownloadRequired downloads only the required models for basic functionality
func (md *ModelDownloader) DownloadRequired() error {
	required := []string{"pigo-facefinder", "openface"}

	fmt.Printf("Downloading required models...\n\n")

	for _, key := range required {
		if err := md.Download(key); err != nil {
			// Try alternative mirrors for OpenFace
			if key == "openface" {
				fmt.Printf("✗ Primary mirror failed, trying alternative...\n")
				if altErr := md.Download("openface-alternative"); altErr != nil {
					fmt.Printf("✗ Alternative mirror failed, trying KDE mirror...\n")
					if kdeErr := md.Download("openface-kde"); kdeErr != nil {
						return fmt.Errorf("all mirrors failed for OpenFace model")
					}
				}
			} else {
				return err
			}
		}
	}

	fmt.Println("\n✓ Required models downloaded successfully")
	return nil
}

// ListAvailableModels lists all available models
func ListAvailableModels() {
	fmt.Println("Available models:")
	fmt.Println()

	for key, model := range AvailableModels {
		fmt.Printf("Key: %s\n", key)
		fmt.Printf("  Name: %s\n", model.Name)
		fmt.Printf("  Description: %s\n", model.Description)
		fmt.Printf("  Size: %s\n", formatBytes(model.Size))
		fmt.Printf("  URL: %s\n", model.URL)
		if model.MD5 != "" {
			fmt.Printf("  MD5: %s\n", model.MD5)
		}
		fmt.Println()
	}
}

// GetModelPath returns the expected path for a downloaded model
func GetModelPath(outputDir, modelKey string) (string, error) {
	model, exists := AvailableModels[modelKey]
	if !exists {
		return "", fmt.Errorf("model '%s' not found", modelKey)
	}

	return filepath.Join(outputDir, model.Filename), nil
}

// Utility functions

// formatBytes formats bytes into human-readable format
func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}

	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}

	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// formatSpeed formats download speed
func formatSpeed(bytesPerSecond float64) string {
	return fmt.Sprintf("%s/s", formatBytes(int64(bytesPerSecond)))
}

// formatDuration formats duration in a human-readable way
func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.0fs", d.Seconds())
	}
	minutes := int(d.Minutes())
	seconds := int(d.Seconds()) % 60
	return fmt.Sprintf("%dm %ds", minutes, seconds)
}

// createHTTPClient creates an HTTP client with proxy support
func (md *ModelDownloader) createHTTPClient() (*http.Client, error) {
	client := &http.Client{
		Timeout: md.Timeout,
	}

	// If proxy URL is provided, configure the client to use it
	if md.ProxyURL != "" {
		proxyURL, err := url.Parse(md.ProxyURL)
		if err != nil {
			return nil, fmt.Errorf("invalid proxy URL: %v", err)
		}

		switch proxyURL.Scheme {
		case "socks5":
			// SOCKS5 proxy
			dialer, err := proxy.SOCKS5("tcp", proxyURL.Host, nil, proxy.Direct)
			if err != nil {
				return nil, fmt.Errorf("failed to create SOCKS5 dialer: %v", err)
			}
			client.Transport = &http.Transport{
				Dial: dialer.Dial,
			}
			fmt.Printf("Using SOCKS5 proxy: %s\n", proxyURL.Host)

		case "http", "https":
			// HTTP/HTTPS proxy
			client.Transport = &http.Transport{
				Proxy: http.ProxyURL(proxyURL),
			}
			fmt.Printf("Using HTTP proxy: %s\n", md.ProxyURL)

		default:
			return nil, fmt.Errorf("unsupported proxy scheme: %s (supported: socks5, http, https)", proxyURL.Scheme)
		}
	}

	return client, nil
}
