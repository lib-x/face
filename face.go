package face

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io/ioutil"
	"math"
	"sync"

	pigo "github.com/esimov/pigo/core"
	"gocv.io/x/gocv"
)

// ModelType defines the face encoding model type
type ModelType string

const (
	// ModelOpenFace is the OpenFace nn4.small2.v1 model (128-dim, 96x96 input)
	ModelOpenFace ModelType = "openface"
	// ModelFaceNet is the FaceNet model (128-dim, 160x160 input)
	ModelFaceNet ModelType = "facenet"
	// ModelArcFace is the ArcFace model (512-dim, 112x112 input)
	ModelArcFace ModelType = "arcface"
	// ModelDlib is the Dlib ResNet model (128-dim, 150x150 input)
	ModelDlib ModelType = "dlib"
	// ModelCustom allows custom model configuration
	ModelCustom ModelType = "custom"
)

// ModelConfig holds model-specific configuration
type ModelConfig struct {
	Type        ModelType
	InputSize   image.Point // Input image size for the model
	FeatureDim  int         // Feature vector dimension
	MeanValues  gocv.Scalar // Mean values for normalization
	ScaleFactor float64     // Scale factor for normalization
	SwapRB      bool        // Swap Red and Blue channels
	Crop        bool        // Center crop
}

// Predefined model configurations
var modelConfigs = map[ModelType]ModelConfig{
	ModelOpenFace: {
		Type:        ModelOpenFace,
		InputSize:   image.Pt(96, 96),
		FeatureDim:  128,
		MeanValues:  gocv.NewScalar(0, 0, 0, 0),
		ScaleFactor: 1.0 / 255.0,
		SwapRB:      true,
		Crop:        false,
	},
	ModelFaceNet: {
		Type:        ModelFaceNet,
		InputSize:   image.Pt(160, 160),
		FeatureDim:  128,
		MeanValues:  gocv.NewScalar(0, 0, 0, 0),
		ScaleFactor: 1.0 / 127.5,
		SwapRB:      true,
		Crop:        false,
	},
	ModelArcFace: {
		Type:        ModelArcFace,
		InputSize:   image.Pt(112, 112),
		FeatureDim:  512,
		MeanValues:  gocv.NewScalar(127.5, 127.5, 127.5, 0),
		ScaleFactor: 1.0 / 127.5,
		SwapRB:      true,
		Crop:        false,
	},
	ModelDlib: {
		Type:        ModelDlib,
		InputSize:   image.Pt(150, 150),
		FeatureDim:  128,
		MeanValues:  gocv.NewScalar(0, 0, 0, 0),
		ScaleFactor: 1.0 / 255.0,
		SwapRB:      true,
		Crop:        false,
	},
}

// FaceFeature represents a face feature vector
type FaceFeature struct {
	PersonID string    `json:"person_id"`
	Feature  []float32 `json:"feature"`
}

// Person represents a person with multiple face samples
type Person struct {
	ID       string        `json:"id"`
	Name     string        `json:"name"`
	Features []FaceFeature `json:"features"`
	mu       sync.RWMutex
}

// RecognizeResult represents a face recognition result
type RecognizeResult struct {
	PersonID    string          `json:"person_id"`
	PersonName  string          `json:"person_name"`
	Confidence  float32         `json:"confidence"`
	BoundingBox image.Rectangle `json:"bounding_box"`
}

// FaceRecognizer is the main face recognition engine
type FaceRecognizer struct {
	pigoClassifier *pigo.Pigo
	faceEncoder    gocv.Net
	modelConfig    ModelConfig
	persons        map[string]*Person
	storage        FaceStorage // Storage backend
	mu             sync.RWMutex
	threshold      float32
	pigoParams     PigoParams
}

// PigoParams holds Pigo face detector parameters
type PigoParams struct {
	MinSize          int     // Minimum face size
	MaxSize          int     // Maximum face size
	ShiftFactor      float64 // Shift factor
	ScaleFactor      float64 // Scale factor
	QualityThreshold float32 // Detection quality threshold
}

// Config holds the basic configuration for FaceRecognizer
type Config struct {
	PigoCascadeFile   string
	FaceEncoderModel  string
	FaceEncoderConfig string // Optional config file for some models
}

// Option is a function that configures FaceRecognizer
type Option func(*FaceRecognizer)

// WithModelType sets the model type (uses predefined configuration)
func WithModelType(modelType ModelType) Option {
	return func(fr *FaceRecognizer) {
		if config, exists := modelConfigs[modelType]; exists {
			fr.modelConfig = config
		}
	}
}

// WithCustomModel sets a custom model configuration
func WithCustomModel(config ModelConfig) Option {
	return func(fr *FaceRecognizer) {
		config.Type = ModelCustom
		fr.modelConfig = config
	}
}

// WithSimilarityThreshold sets the similarity threshold for recognition
func WithSimilarityThreshold(threshold float32) Option {
	return func(fr *FaceRecognizer) {
		fr.threshold = threshold
	}
}

// WithPigoParams sets custom Pigo detector parameters
func WithPigoParams(params PigoParams) Option {
	return func(fr *FaceRecognizer) {
		fr.pigoParams = params
	}
}

// WithMinFaceSize sets the minimum face size for detection
func WithMinFaceSize(size int) Option {
	return func(fr *FaceRecognizer) {
		fr.pigoParams.MinSize = size
	}
}

// WithMaxFaceSize sets the maximum face size for detection
func WithMaxFaceSize(size int) Option {
	return func(fr *FaceRecognizer) {
		fr.pigoParams.MaxSize = size
	}
}

// WithStorage sets a custom storage backend
func WithStorage(storage FaceStorage) Option {
	return func(fr *FaceRecognizer) {
		fr.storage = storage
	}
}

// NewFaceRecognizer creates a new FaceRecognizer instance
func NewFaceRecognizer(config Config, opts ...Option) (*FaceRecognizer, error) {
	fr := &FaceRecognizer{
		persons:   make(map[string]*Person),
		storage:   NewMemoryStorage(), // Default to memory storage
		threshold: 0.6,                // Default threshold
		pigoParams: PigoParams{
			MinSize:          100,
			MaxSize:          1000,
			ShiftFactor:      0.1,
			ScaleFactor:      1.1,
			QualityThreshold: 5.0,
		},
		modelConfig: modelConfigs[ModelOpenFace], // Default model
	}

	// Apply options
	for _, opt := range opts {
		opt(fr)
	}

	// Load Pigo face detector
	cascadeFile, err := ioutil.ReadFile(config.PigoCascadeFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read Pigo cascade file: %v", err)
	}

	p := pigo.NewPigo()
	classifier, err := p.Unpack(cascadeFile)
	if err != nil {
		return nil, fmt.Errorf("failed to unpack Pigo cascade: %v", err)
	}
	fr.pigoClassifier = classifier

	// Load face encoder model
	if config.FaceEncoderConfig != "" {
		fr.faceEncoder = gocv.ReadNet(config.FaceEncoderModel, config.FaceEncoderConfig)
	} else {
		fr.faceEncoder = gocv.ReadNet(config.FaceEncoderModel, "")
	}

	if fr.faceEncoder.Empty() {
		return nil, errors.New("failed to load face encoder model")
	}

	return fr, nil
}

// Close releases all resources
func (fr *FaceRecognizer) Close() error {
	if !fr.faceEncoder.Empty() {
		return fr.faceEncoder.Close()
	}
	return nil
}

// DetectFaces detects faces in an image using Pigo
func (fr *FaceRecognizer) DetectFaces(img image.Image) []image.Rectangle {
	// Convert to grayscale
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	pixels := make([]uint8, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Convert to grayscale using luminosity method
			gray := uint8((r*299 + g*587 + b*114) / 1000 / 256)
			pixels[y*width+x] = gray
		}
	}

	// Pigo detection parameters
	cParams := pigo.CascadeParams{
		MinSize:     fr.pigoParams.MinSize,
		MaxSize:     fr.pigoParams.MaxSize,
		ShiftFactor: fr.pigoParams.ShiftFactor,
		ScaleFactor: fr.pigoParams.ScaleFactor,
		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   height,
			Cols:   width,
			Dim:    width,
		},
	}

	// Run cascade detector
	dets := fr.pigoClassifier.RunCascade(cParams, 0.0)
	dets = fr.pigoClassifier.ClusterDetections(dets, 0.2)

	// Convert to image.Rectangle
	faces := make([]image.Rectangle, 0, len(dets))
	for _, det := range dets {
		if det.Q > fr.pigoParams.QualityThreshold {
			x := det.Col - det.Scale/2
			y := det.Row - det.Scale/2
			faces = append(faces, image.Rect(x, y, x+det.Scale, y+det.Scale))
		}
	}

	return faces
}

// ExtractFeature extracts face feature vector using the configured model
func (fr *FaceRecognizer) ExtractFeature(faceImg gocv.Mat) ([]float32, error) {
	if faceImg.Empty() {
		return nil, errors.New("input image is empty")
	}

	// Resize to model's input size
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(faceImg, &resized, fr.modelConfig.InputSize, 0, 0, gocv.InterpolationLinear)

	// Create blob with model-specific parameters
	blob := gocv.BlobFromImage(
		resized,
		fr.modelConfig.ScaleFactor,
		fr.modelConfig.InputSize,
		fr.modelConfig.MeanValues,
		fr.modelConfig.SwapRB,
		fr.modelConfig.Crop,
	)
	defer blob.Close()

	// Forward pass
	fr.faceEncoder.SetInput(blob, "")
	output := fr.faceEncoder.Forward("")
	defer output.Close()

	// Convert to float32 slice
	feature := make([]float32, output.Total())
	for i := 0; i < output.Total(); i++ {
		feature[i] = output.GetFloatAt(0, i)
	}

	// L2 normalization
	return normalizeFeature(feature), nil
}

// AddPerson adds a new person to the recognition database
func (fr *FaceRecognizer) AddPerson(id, name string) error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if _, exists := fr.persons[id]; exists {
		return fmt.Errorf("person ID %s already exists", id)
	}

	fr.persons[id] = &Person{
		ID:       id,
		Name:     name,
		Features: make([]FaceFeature, 0),
	}

	return nil
}

// AddFaceSample adds a face sample for a specific person
func (fr *FaceRecognizer) AddFaceSample(personID string, img gocv.Mat) error {
	fr.mu.RLock()
	person, exists := fr.persons[personID]
	fr.mu.RUnlock()

	if !exists {
		return fmt.Errorf("person ID %s does not exist", personID)
	}

	// Detect faces
	goImg, err := img.ToImage()
	if err != nil {
		return fmt.Errorf("failed to convert image: %v", err)
	}

	faces := fr.DetectFaces(goImg)
	if len(faces) == 0 {
		return errors.New("no face detected in image")
	}

	// Use the first detected face
	faceRegion := img.Region(faces[0])
	defer faceRegion.Close()

	// Extract feature
	feature, err := fr.ExtractFeature(faceRegion)
	if err != nil {
		return fmt.Errorf("failed to extract feature: %v", err)
	}

	// Add feature to person
	person.mu.Lock()
	person.Features = append(person.Features, FaceFeature{
		PersonID: personID,
		Feature:  feature,
	})
	person.mu.Unlock()

	return nil
}

// Recognize recognizes faces in an image
func (fr *FaceRecognizer) Recognize(img gocv.Mat) ([]RecognizeResult, error) {
	// Detect faces
	goImg, err := img.ToImage()
	if err != nil {
		return nil, fmt.Errorf("failed to convert image: %v", err)
	}

	faces := fr.DetectFaces(goImg)
	if len(faces) == 0 {
		return []RecognizeResult{}, nil
	}

	results := make([]RecognizeResult, 0, len(faces))

	// Recognize each detected face
	for _, faceRect := range faces {
		faceRegion := img.Region(faceRect)
		feature, err := fr.ExtractFeature(faceRegion)
		faceRegion.Close()

		if err != nil {
			continue
		}

		// Match person
		personID, personName, confidence := fr.matchPerson(feature)

		if confidence >= fr.threshold {
			results = append(results, RecognizeResult{
				PersonID:    personID,
				PersonName:  personName,
				Confidence:  confidence,
				BoundingBox: faceRect,
			})
		} else {
			results = append(results, RecognizeResult{
				PersonID:    "unknown",
				PersonName:  "Unknown",
				Confidence:  confidence,
				BoundingBox: faceRect,
			})
		}
	}

	return results, nil
}

// matchPerson finds the best matching person for a feature vector
func (fr *FaceRecognizer) matchPerson(feature []float32) (string, string, float32) {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	var bestPersonID, bestPersonName string
	var bestConfidence float32 = 0

	for _, person := range fr.persons {
		person.mu.RLock()
		for _, sample := range person.Features {
			similarity := cosineSimilarity(feature, sample.Feature)
			if similarity > bestConfidence {
				bestConfidence = similarity
				bestPersonID = person.ID
				bestPersonName = person.Name
			}
		}
		person.mu.RUnlock()
	}

	return bestPersonID, bestPersonName, bestConfidence
}

// GetPerson retrieves a person by ID
func (fr *FaceRecognizer) GetPerson(id string) (*Person, error) {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	person, exists := fr.persons[id]
	if !exists {
		return nil, fmt.Errorf("person ID %s does not exist", id)
	}

	return person, nil
}

// ListPersons returns all registered persons
func (fr *FaceRecognizer) ListPersons() []*Person {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	persons := make([]*Person, 0, len(fr.persons))
	for _, person := range fr.persons {
		persons = append(persons, person)
	}

	return persons
}

// RemovePerson removes a person from the database
func (fr *FaceRecognizer) RemovePerson(id string) error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if _, exists := fr.persons[id]; !exists {
		return fmt.Errorf("person ID %s does not exist", id)
	}

	delete(fr.persons, id)
	return nil
}

// SaveDatabase saves the face database to a JSON file
func (fr *FaceRecognizer) SaveDatabase(filepath string) error {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	data, err := json.MarshalIndent(fr.persons, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal database: %v", err)
	}

	return ioutil.WriteFile(filepath, data, 0644)
}

// LoadDatabase loads the face database from a JSON file
func (fr *FaceRecognizer) LoadDatabase(filepath string) error {
	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("failed to read database file: %v", err)
	}

	persons := make(map[string]*Person)
	if err := json.Unmarshal(data, &persons); err != nil {
		return fmt.Errorf("failed to unmarshal database: %v", err)
	}

	fr.mu.Lock()
	fr.persons = persons
	fr.mu.Unlock()

	return nil
}

// SetThreshold sets the similarity threshold
func (fr *FaceRecognizer) SetThreshold(threshold float32) {
	fr.threshold = threshold
}

// GetThreshold returns the current similarity threshold
func (fr *FaceRecognizer) GetThreshold() float32 {
	return fr.threshold
}

// GetModelConfig returns the current model configuration
func (fr *FaceRecognizer) GetModelConfig() ModelConfig {
	return fr.modelConfig
}

// GetStorage returns the storage backend
func (fr *FaceRecognizer) GetStorage() FaceStorage {
	fr.mu.RLock()
	defer fr.mu.RUnlock()
	return fr.storage
}

// GetSampleCount returns the number of samples for a person
func (fr *FaceRecognizer) GetSampleCount(personID string) (int, error) {
	fr.mu.RLock()
	person, exists := fr.persons[personID]
	fr.mu.RUnlock()

	if !exists {
		return 0, fmt.Errorf("person ID %s does not exist", personID)
	}

	person.mu.RLock()
	count := len(person.Features)
	person.mu.RUnlock()

	return count, nil
}

// Utility functions

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// euclideanDistance calculates the Euclidean distance between two vectors
func euclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.MaxFloat32)
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// normalizeFeature performs L2 normalization on a feature vector
func normalizeFeature(feature []float32) []float32 {
	var norm float32
	for _, v := range feature {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm == 0 {
		return feature
	}

	normalized := make([]float32, len(feature))
	for i, v := range feature {
		normalized[i] = v / norm
	}

	return normalized
}
