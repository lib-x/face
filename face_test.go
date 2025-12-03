package face

import (
	"image"
	"image/color"
	"math"
	"os"
	"testing"

	"gocv.io/x/gocv"
)

// Test helpers

// createTestImage creates a test image with a simple face-like circle
func createTestImage(width, height int) gocv.Mat {
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)

	// Fill with white background
	white := color.RGBA{255, 255, 255, 255}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetUCharAt(y, x*3, white.B)
			img.SetUCharAt(y, x*3+1, white.G)
			img.SetUCharAt(y, x*3+2, white.R)
		}
	}

	// Draw a simple "face" (circle)
	center := image.Pt(width/2, height/2)
	gocv.Circle(&img, center, width/3, color.RGBA{200, 200, 200, 255}, -1)

	return img
}

// skipIfModelsNotAvailable skips test if model files are not available
func skipIfModelsNotAvailable(t *testing.T) {
	if _, err := os.Stat("./testdata/facefinder"); os.IsNotExist(err) {
		t.Skip("Model files not available (run in testdata directory or download models)")
	}
}

// Test: FaceRecognizer initialization

func TestNewFaceRecognizer_DefaultOptions(t *testing.T) {
	skipIfModelsNotAvailable(t)

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	// Verify default values
	if recognizer.GetThreshold() != 0.6 {
		t.Errorf("Expected default threshold 0.6, got %.2f", recognizer.GetThreshold())
	}

	modelConfig := recognizer.GetModelConfig()
	if modelConfig.Type != ModelOpenFace {
		t.Errorf("Expected default model OpenFace, got %s", modelConfig.Type)
	}
}

func TestNewFaceRecognizer_WithOptions(t *testing.T) {
	skipIfModelsNotAvailable(t)

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(
		config,
		WithModelType(ModelOpenFace),
		WithSimilarityThreshold(0.7),
		WithMinFaceSize(80),
		WithMaxFaceSize(800),
	)
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	// Verify options applied
	if recognizer.GetThreshold() != 0.7 {
		t.Errorf("Expected threshold 0.7, got %.2f", recognizer.GetThreshold())
	}

	if recognizer.pigoParams.MinSize != 80 {
		t.Errorf("Expected MinSize 80, got %d", recognizer.pigoParams.MinSize)
	}

	if recognizer.pigoParams.MaxSize != 800 {
		t.Errorf("Expected MaxSize 800, got %d", recognizer.pigoParams.MaxSize)
	}
}

func TestNewFaceRecognizer_CustomModelConfig(t *testing.T) {
	skipIfModelsNotAvailable(t)

	customModel := ModelConfig{
		InputSize:   image.Pt(128, 128),
		FeatureDim:  256,
		MeanValues:  gocv.NewScalar(127.5, 127.5, 127.5, 0),
		ScaleFactor: 1.0 / 127.5,
		SwapRB:      true,
		Crop:        false,
	}

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config, WithCustomModel(customModel))
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	modelConfig := recognizer.GetModelConfig()
	if modelConfig.Type != ModelCustom {
		t.Errorf("Expected custom model type, got %s", modelConfig.Type)
	}

	if modelConfig.InputSize.X != 128 || modelConfig.InputSize.Y != 128 {
		t.Errorf("Expected input size 128x128, got %dx%d",
			modelConfig.InputSize.X, modelConfig.InputSize.Y)
	}

	if modelConfig.FeatureDim != 256 {
		t.Errorf("Expected feature dim 256, got %d", modelConfig.FeatureDim)
	}
}

// Test: Person management

func TestAddPerson(t *testing.T) {
	skipIfModelsNotAvailable(t)

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	// Test adding person
	err = recognizer.AddPerson("001", "Alice")
	if err != nil {
		t.Fatalf("Failed to add person: %v", err)
	}

	// Test duplicate ID
	err = recognizer.AddPerson("001", "Bob")
	if err == nil {
		t.Error("Expected error when adding duplicate ID, got nil")
	}

	// Verify person was added
	person, err := recognizer.GetPerson("001")
	if err != nil {
		t.Fatalf("Failed to get person: %v", err)
	}

	if person.Name != "Alice" {
		t.Errorf("Expected name 'Alice', got '%s'", person.Name)
	}

	if person.ID != "001" {
		t.Errorf("Expected ID '001', got '%s'", person.ID)
	}
}

func TestRemovePerson(t *testing.T) {
	skipIfModelsNotAvailable(t)

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	// Add and remove person
	recognizer.AddPerson("001", "Alice")

	err = recognizer.RemovePerson("001")
	if err != nil {
		t.Fatalf("Failed to remove person: %v", err)
	}

	// Verify person was removed
	_, err = recognizer.GetPerson("001")
	if err == nil {
		t.Error("Expected error when getting removed person, got nil")
	}

	// Test removing non-existent person
	err = recognizer.RemovePerson("999")
	if err == nil {
		t.Error("Expected error when removing non-existent person, got nil")
	}
}

func TestListPersons(t *testing.T) {
	skipIfModelsNotAvailable(t)

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	// Add multiple persons
	persons := []struct {
		id   string
		name string
	}{
		{"001", "Alice"},
		{"002", "Bob"},
		{"003", "Charlie"},
	}

	for _, p := range persons {
		recognizer.AddPerson(p.id, p.name)
	}

	// List persons
	list := recognizer.ListPersons()
	if len(list) != 3 {
		t.Errorf("Expected 3 persons, got %d", len(list))
	}

	// Verify all persons are in the list
	found := make(map[string]bool)
	for _, p := range list {
		found[p.ID] = true
	}

	for _, p := range persons {
		if !found[p.id] {
			t.Errorf("Person %s not found in list", p.id)
		}
	}
}

func TestGetSampleCount(t *testing.T) {
	skipIfModelsNotAvailable(t)

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	// Add person
	recognizer.AddPerson("001", "Alice")

	// Initial sample count should be 0
	count, err := recognizer.GetSampleCount("001")
	if err != nil {
		t.Fatalf("Failed to get sample count: %v", err)
	}
	if count != 0 {
		t.Errorf("Expected initial sample count 0, got %d", count)
	}

	// Test non-existent person
	_, err = recognizer.GetSampleCount("999")
	if err == nil {
		t.Error("Expected error for non-existent person, got nil")
	}
}

// Test: Threshold management

func TestSetGetThreshold(t *testing.T) {
	skipIfModelsNotAvailable(t)

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	// Test default threshold
	defaultThreshold := recognizer.GetThreshold()
	if defaultThreshold != 0.6 {
		t.Errorf("Expected default threshold 0.6, got %.2f", defaultThreshold)
	}

	// Test setting new threshold
	testThresholds := []float32{0.5, 0.7, 0.8}
	for _, threshold := range testThresholds {
		recognizer.SetThreshold(threshold)
		got := recognizer.GetThreshold()
		if got != threshold {
			t.Errorf("Expected threshold %.2f, got %.2f", threshold, got)
		}
	}
}

// Test: Utility functions

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name      string
		a         []float32
		b         []float32
		expected  float32
		tolerance float32
	}{
		{
			name:      "Identical vectors",
			a:         []float32{1, 0, 0},
			b:         []float32{1, 0, 0},
			expected:  1.0,
			tolerance: 0.0001,
		},
		{
			name:      "Orthogonal vectors",
			a:         []float32{1, 0, 0},
			b:         []float32{0, 1, 0},
			expected:  0.0,
			tolerance: 0.0001,
		},
		{
			name:      "Opposite vectors",
			a:         []float32{1, 0, 0},
			b:         []float32{-1, 0, 0},
			expected:  -1.0,
			tolerance: 0.0001,
		},
		{
			name:      "45-degree angle",
			a:         []float32{1, 0},
			b:         []float32{1, 1},
			expected:  0.7071, // cos(45°) ≈ 0.7071
			tolerance: 0.001,
		},
		{
			name:      "Different length vectors",
			a:         []float32{1, 0},
			b:         []float32{1, 0, 0},
			expected:  0.0, // Should return 0 for different lengths
			tolerance: 0.0001,
		},
		{
			name:      "Empty vectors",
			a:         []float32{},
			b:         []float32{},
			expected:  0.0,
			tolerance: 0.0001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)

			if math.Abs(float64(result-tt.expected)) > float64(tt.tolerance) {
				t.Errorf("Expected %.4f, got %.4f (tolerance: %.4f)",
					tt.expected, result, tt.tolerance)
			}
		})
	}
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name      string
		a         []float32
		b         []float32
		expected  float32
		tolerance float32
	}{
		{
			name:      "Identical vectors",
			a:         []float32{1, 2, 3},
			b:         []float32{1, 2, 3},
			expected:  0.0,
			tolerance: 0.0001,
		},
		{
			name:      "Unit distance",
			a:         []float32{0, 0},
			b:         []float32{1, 0},
			expected:  1.0,
			tolerance: 0.0001,
		},
		{
			name:      "3-4-5 triangle",
			a:         []float32{0, 0},
			b:         []float32{3, 4},
			expected:  5.0,
			tolerance: 0.0001,
		},
		{
			name:      "Different length vectors",
			a:         []float32{1, 2},
			b:         []float32{1, 2, 3},
			expected:  float32(math.MaxFloat32),
			tolerance: 0.0001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := euclideanDistance(tt.a, tt.b)

			if math.Abs(float64(result-tt.expected)) > float64(tt.tolerance) {
				t.Errorf("Expected %.4f, got %.4f", tt.expected, result)
			}
		})
	}
}

func TestNormalizeFeature(t *testing.T) {
	tests := []struct {
		name      string
		input     []float32
		expectLen float32 // Expected L2 norm (should be 1.0 after normalization)
	}{
		{
			name:      "Simple vector",
			input:     []float32{3, 4},
			expectLen: 1.0,
		},
		{
			name:      "Already normalized",
			input:     []float32{1, 0, 0},
			expectLen: 1.0,
		},
		{
			name:      "Multi-dimensional",
			input:     []float32{1, 2, 3, 4},
			expectLen: 1.0,
		},
		{
			name:      "Negative values",
			input:     []float32{-3, 4},
			expectLen: 1.0,
		},
		{
			name:      "Zero vector",
			input:     []float32{0, 0, 0},
			expectLen: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			normalized := normalizeFeature(tt.input)

			// Calculate L2 norm
			var sumSquares float32
			for _, v := range normalized {
				sumSquares += v * v
			}
			length := float32(math.Sqrt(float64(sumSquares)))

			tolerance := float32(0.0001)
			if math.Abs(float64(length-tt.expectLen)) > float64(tolerance) {
				t.Errorf("Expected L2 norm %.4f, got %.4f", tt.expectLen, length)
			}
		})
	}
}

// Test: Model configuration

func TestModelConfigs(t *testing.T) {
	tests := []struct {
		modelType    ModelType
		expectedSize image.Point
		expectedDim  int
	}{
		{ModelOpenFace, image.Pt(96, 96), 128},
		{ModelFaceNet, image.Pt(160, 160), 128},
		{ModelArcFace, image.Pt(112, 112), 512},
		{ModelDlib, image.Pt(150, 150), 128},
	}

	for _, tt := range tests {
		t.Run(string(tt.modelType), func(t *testing.T) {
			config, exists := modelConfigs[tt.modelType]
			if !exists {
				t.Fatalf("Model config for %s not found", tt.modelType)
			}

			if config.InputSize != tt.expectedSize {
				t.Errorf("Expected input size %v, got %v",
					tt.expectedSize, config.InputSize)
			}

			if config.FeatureDim != tt.expectedDim {
				t.Errorf("Expected feature dimension %d, got %d",
					tt.expectedDim, config.FeatureDim)
			}
		})
	}
}

// Benchmark tests

func BenchmarkCosineSimilarity(b *testing.B) {
	a := make([]float32, 128)
	vec := make([]float32, 128)

	for i := range a {
		a[i] = float32(i)
		vec[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cosineSimilarity(a, vec)
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	a := make([]float32, 128)
	vec := make([]float32, 128)

	for i := range a {
		a[i] = float32(i)
		vec[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		euclideanDistance(a, vec)
	}
}

func BenchmarkNormalizeFeature(b *testing.B) {
	feature := make([]float32, 128)
	for i := range feature {
		feature[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		normalizeFeature(feature)
	}
}

func BenchmarkDetectFaces(b *testing.B) {
	// Skip if models not available
	if _, err := os.Stat("./testdata/facefinder"); os.IsNotExist(err) {
		b.Skip("Model files not available")
	}

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		b.Skipf("Failed to initialize recognizer: %v", err)
	}
	defer recognizer.Close()

	testImg := createTestImage(640, 480)
	defer testImg.Close()

	goImg, _ := testImg.ToImage()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		recognizer.DetectFaces(goImg)
	}
}

func BenchmarkExtractFeature(b *testing.B) {
	// Skip if models not available
	if _, err := os.Stat("./testdata/facefinder"); os.IsNotExist(err) {
		b.Skip("Model files not available")
	}

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		b.Skipf("Failed to initialize recognizer: %v", err)
	}
	defer recognizer.Close()

	testImg := createTestImage(200, 200)
	defer testImg.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		recognizer.ExtractFeature(testImg)
	}
}

// Table-driven tests for comprehensive coverage

func TestAddPerson_TableDriven(t *testing.T) {
	skipIfModelsNotAvailable(t)

	config := Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	tests := []struct {
		name        string
		personID    string
		personName  string
		expectError bool
	}{
		{"Valid person", "001", "Alice", false},
		{"Another valid person", "002", "Bob", false},
		{"Empty name", "003", "", false}, // Empty name is allowed
		{"Duplicate ID", "001", "Charlie", true},
		{"Special characters in name", "004", "José García", false},
		{"Long ID", "123456789012345678901234567890", "Dave", false},
	}

	recognizer, err := NewFaceRecognizer(config)
	if err != nil {
		t.Skipf("Skip test (model files not available): %v", err)
		return
	}
	defer recognizer.Close()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := recognizer.AddPerson(tt.personID, tt.personName)

			if tt.expectError && err == nil {
				t.Error("Expected error but got nil")
			}

			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}
