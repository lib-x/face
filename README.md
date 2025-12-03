# Go Face Recognizer

A high-performance face recognition library for Go, combining Pigo (fast detection) and GoCV (feature extraction), with support for multiple deep learning models and multi-sample recognition.

## Features

✅ **Fast Detection**: Efficient face detection using Pigo  
✅ **Multiple Models**: Support for OpenFace, FaceNet, ArcFace, Dlib, and custom models  
✅ **Flexible Configuration**: Options pattern for easy customization  
✅ **Multi-Sample Training**: Register multiple photos per person for improved accuracy  
✅ **Persistent Storage**: Save/load face database in JSON format  
✅ **Thread-Safe**: Built-in concurrency control for multi-threaded environments  
✅ **Easy to Use**: Clean and intuitive API design  

## Dependencies

### Go Packages
```bash
go get -u gocv.io/x/gocv
go get -u github.com/esimov/pigo/core
```

### System Dependencies

**OpenCV** (required for GoCV):
```bash
# macOS
brew install opencv

# Ubuntu/Debian
sudo apt-get install libopencv-dev

# For detailed installation, see GoCV documentation
# https://gocv.io/getting-started/
```

## Model Files

### Automatic Download (Recommended)

Use the built-in model downloader:

```bash
# Download required models (Pigo + OpenFace)
go run cmd/download-models/main.go

# List available models
go run cmd/download-models/main.go -list

# Download specific model
go run cmd/download-models/main.go -model=openface

# Download all available models
go run cmd/download-models/main.go -all

# Custom output directory
go run cmd/download-models/main.go -output=/path/to/models
```

Or use the downloader in your code:

```go
import fr "github.com/lib-x/face"

// Download required models
downloader := fr.NewModelDownloader("./models")
if err := downloader.DownloadRequired(); err != nil {
    log.Fatal(err)
}

// Download specific model
if err := downloader.Download("openface"); err != nil {
    log.Fatal(err)
}
```
### Toolchains
#### Arch User
```bash
paru -S  hdf5 vtk opencv opencv-samples base-devel pkgconf
```
### Manual Download

If you prefer manual download:

1. **Pigo Face Detector**
```bash
mkdir -p models
wget https://raw.githubusercontent.com/esimov/pigo/master/cascade/facefinder -O models/facefinder
```

2. **OpenFace Model** (96x96 input, 128-dim, fastest)
```bash
# Primary mirror
wget https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7 -O models/nn4.small2.v1.t7

# Alternative mirrors if primary fails
wget https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7 -O models/nn4.small2.v1.t7
# OR
wget https://files.kde.org/digikam/facesengine/dnnface/openface_nn4.small2.v1.t7 -O models/nn4.small2.v1.t7
```

3. **Other Models** (optional):

**FaceNet** (160x160 input, 128-dim, balanced)
```bash
# Download from TensorFlow models repository
# See: https://github.com/davidsandberg/facenet
```

**ArcFace** (112x112 input, 512-dim, most accurate)
```bash
# Download ONNX model
# See: https://github.com/onnx/models/tree/main/vision/body_analysis/arcface
```

### Available Models

| Model | Auto-Download | Size | MD5 Checksum |
|-------|--------------|------|--------------|
| Pigo facefinder | ✅ | ~50KB | N/A |
| OpenFace nn4.small2.v1 | ✅ | ~30MB | c95bfd8cc1adf05210e979ff623013b6 |
| FaceNet | ❌ | Variable | Manual |
| ArcFace | ❌ | Variable | Manual |

## Quick Start

### 1. Basic Usage with Default Model

```go
package main

import (
    "fmt"
    fr "github.com/lib-x/face"
    "gocv.io/x/gocv"
)

func main() {
    // Initialize with default OpenFace model
    recognizer, err := fr.NewFaceRecognizer(
        fr.Config{
            PigoCascadeFile:  "./models/facefinder",
            FaceEncoderModel: "./models/nn4.small2.v1.t7",
        },
    )
    if err != nil {
        panic(err)
    }
    defer recognizer.Close()
    
    // Add a person
    recognizer.AddPerson("001", "Alice")
    
    // Add face samples
    img := gocv.IMRead("alice.jpg", gocv.IMReadColor)
    recognizer.AddFaceSample("001", img)
    img.Close()
    
    // Recognize faces
    testImg := gocv.IMRead("test.jpg", gocv.IMReadColor)
    results, _ := recognizer.Recognize(testImg)
    
    for _, r := range results {
        fmt.Printf("Found: %s (%.2f%%)\n", r.PersonName, r.Confidence*100)
    }
}
```

### 2. Using Different Models

```go
// OpenFace (fastest, good accuracy)
recognizer, _ := fr.NewFaceRecognizer(
    fr.Config{
        PigoCascadeFile:  "./models/facefinder",
        FaceEncoderModel: "./models/nn4.small2.v1.t7",
    },
    fr.WithModelType(fr.ModelOpenFace),
    fr.WithSimilarityThreshold(0.6),
)

// FaceNet (balanced)
recognizer, _ := fr.NewFaceRecognizer(
    fr.Config{
        PigoCascadeFile:   "./models/facefinder",
        FaceEncoderModel:  "./models/facenet.pb",
        FaceEncoderConfig: "./models/facenet.pbtxt",
    },
    fr.WithModelType(fr.ModelFaceNet),
    fr.WithSimilarityThreshold(0.65),
)

// ArcFace (most accurate)
recognizer, _ := fr.NewFaceRecognizer(
    fr.Config{
        PigoCascadeFile:  "./models/facefinder",
        FaceEncoderModel: "./models/arcface.onnx",
    },
    fr.WithModelType(fr.ModelArcFace),
    fr.WithSimilarityThreshold(0.7),
)
```

### 3. Custom Configuration

```go
recognizer, _ := fr.NewFaceRecognizer(
    fr.Config{
        PigoCascadeFile:  "./models/facefinder",
        FaceEncoderModel: "./models/nn4.small2.v1.t7",
    },
    // Set model type
    fr.WithModelType(fr.ModelOpenFace),
    // Set similarity threshold
    fr.WithSimilarityThreshold(0.65),
    // Set face detection parameters
    fr.WithMinFaceSize(80),
    fr.WithMaxFaceSize(800),
    // Or set all Pigo parameters at once
    fr.WithPigoParams(fr.PigoParams{
        MinSize:          80,
        MaxSize:          800,
        ShiftFactor:      0.1,
        ScaleFactor:      1.1,
        QualityThreshold: 6.0,
    }),
)
```

### 4. Custom Model Configuration

```go
customModel := fr.ModelConfig{
    InputSize:   image.Pt(128, 128),
    FeatureDim:  256,
    MeanValues:  gocv.NewScalar(127.5, 127.5, 127.5, 0),
    ScaleFactor: 1.0 / 127.5,
    SwapRB:      true,
    Crop:        false,
}

recognizer, _ := fr.NewFaceRecognizer(
    fr.Config{
        PigoCascadeFile:  "./models/facefinder",
        FaceEncoderModel: "./models/custom.onnx",
    },
    fr.WithCustomModel(customModel),
)
```

## Supported Model Types

| Model | Input Size | Features | Speed | Accuracy | Threshold |
|-------|-----------|----------|-------|----------|-----------|
| OpenFace | 96x96 | 128-dim | ⚡⚡⚡ | ⭐⭐⭐ | 0.6 |
| FaceNet | 160x160 | 128-dim | ⚡⚡ | ⭐⭐⭐⭐ | 0.65 |
| ArcFace | 112x112 | 512-dim | ⚡ | ⭐⭐⭐⭐⭐ | 0.7 |
| Dlib | 150x150 | 128-dim | ⚡⚡ | ⭐⭐⭐⭐ | 0.6 |
| Custom | Variable | Variable | - | - | Adjust |

## API Documentation

### Initialization Options

```go
// WithModelType sets predefined model configuration
func WithModelType(modelType ModelType) Option

// WithCustomModel sets custom model configuration
func WithCustomModel(config ModelConfig) Option

// WithSimilarityThreshold sets recognition threshold (0.0-1.0)
func WithSimilarityThreshold(threshold float32) Option

// WithPigoParams sets Pigo detector parameters
func WithPigoParams(params PigoParams) Option

// WithMinFaceSize sets minimum face size for detection
func WithMinFaceSize(size int) Option

// WithMaxFaceSize sets maximum face size for detection
func WithMaxFaceSize(size int) Option
```

### Person Management

```go
// Add a new person
func (fr *FaceRecognizer) AddPerson(id, name string) error

// Remove a person
func (fr *FaceRecognizer) RemovePerson(id string) error

// Get person information
func (fr *FaceRecognizer) GetPerson(id string) (*Person, error)

// List all persons
func (fr *FaceRecognizer) ListPersons() []*Person

// Get sample count for a person
func (fr *FaceRecognizer) GetSampleCount(personID string) (int, error)
```

### Face Recognition

```go
// Add a face sample for a person
func (fr *FaceRecognizer) AddFaceSample(personID string, img gocv.Mat) error

// Recognize faces in an image
func (fr *FaceRecognizer) Recognize(img gocv.Mat) ([]RecognizeResult, error)

// Detect faces (detection only, no recognition)
func (fr *FaceRecognizer) DetectFaces(img image.Image) []image.Rectangle

// Extract feature vector from face image
func (fr *FaceRecognizer) ExtractFeature(faceImg gocv.Mat) ([]float32, error)
```

### Database Operations

```go
// Save database to JSON file
func (fr *FaceRecognizer) SaveDatabase(filepath string) error

// Load database from JSON file
func (fr *FaceRecognizer) LoadDatabase(filepath string) error
```

### Configuration

```go
// Set similarity threshold
func (fr *FaceRecognizer) SetThreshold(threshold float32)

// Get current threshold
func (fr *FaceRecognizer) GetThreshold() float32

// Get model configuration
func (fr *FaceRecognizer) GetModelConfig() ModelConfig
```

## Model Configuration Structure

```go
type ModelConfig struct {
    Type        ModelType    // Model type identifier
    InputSize   image.Point  // Model input size (width, height)
    FeatureDim  int          // Feature vector dimension
    MeanValues  gocv.Scalar  // Mean values for normalization
    ScaleFactor float64      // Scale factor for normalization
    SwapRB      bool         // Swap Red and Blue channels
    Crop        bool         // Center crop input image
}
```

## Recognition Result

```go
type RecognizeResult struct {
    PersonID    string          // Person identifier
    PersonName  string          // Person name
    Confidence  float32         // Confidence score (0.0-1.0)
    BoundingBox image.Rectangle // Face bounding box
}
```

## Best Practices

### 1. Sample Collection
- **Quantity**: 3-5 photos per person
- **Variety**: Different angles (front, left, right)
- **Expressions**: Neutral, smiling, different expressions
- **Lighting**: Various lighting conditions
- **Quality**: Clear, high-resolution images

### 2. Threshold Selection

**Strict Mode** (0.7-0.8):
- High precision, may miss some matches
- Use when false positives are costly

**Balanced Mode** (0.6-0.7):
- Recommended for most applications
- Good balance between precision and recall

**Relaxed Mode** (0.5-0.6):
- High recall, may have false positives
- Use when missing matches is more costly

### 3. Model Selection

**Use OpenFace when**:
- Speed is critical
- Running on limited hardware
- Good accuracy is sufficient

**Use FaceNet when**:
- Need balanced performance
- Moderate hardware available
- Better accuracy required

**Use ArcFace when**:
- Maximum accuracy needed
- GPU available
- Speed is less critical

### 4. Image Quality Requirements
- **Resolution**: Minimum 640x480
- **Face Size**: At least 100x100 pixels
- **Lighting**: Avoid overexposure or underexposure
- **Clarity**: Avoid motion blur
- **Angle**: Frontal or near-frontal faces work best

### 5. Performance Optimization

```go
// Concurrent processing
var wg sync.WaitGroup
results := make(chan []fr.RecognizeResult, len(images))

for _, imgPath := range images {
    wg.Add(1)
    go func(path string) {
        defer wg.Done()
        img := gocv.IMRead(path, gocv.IMReadColor)
        defer img.Close()
        
        res, _ := recognizer.Recognize(img)
        results <- res
    }(imgPath)
}

wg.Wait()
close(results)
```

## Project Structure

```
your-project/
├── models/
│   ├── facefinder              # Pigo detector
│   ├── nn4.small2.v1.t7        # OpenFace model
│   ├── facenet.pb              # FaceNet model (optional)
│   └── arcface.onnx            # ArcFace model (optional)
├── images/
│   ├── person1_sample1.jpg
│   ├── person1_sample2.jpg
│   └── ...
├── test_images/
│   └── test.jpg
├── output/
│   └── results/
├── face_database.json          # Face database
└── main.go
```

## Troubleshooting

### Q: Low recognition accuracy?
A:
1. Add more samples per person (3-5 recommended)
2. Ensure sample quality (clear, well-lit, frontal)
3. Adjust similarity threshold
4. Try a different model (FaceNet or ArcFace)
5. Verify model files are loaded correctly

### Q: Face not detected?
A:
1. Ensure face size is at least 100x100 pixels
2. Check image quality and lighting
3. Adjust Pigo's MinSize and MaxSize parameters
4. Verify face is frontal or near-frontal

### Q: What image formats are supported?
A: All formats supported by GoCV: JPG, PNG, BMP, TIFF, etc.

### Q: Can this work with real-time video?
A: Yes, but consider:
- Reduce frame rate (e.g., process every 5th frame)
- Use faster model (OpenFace)
- Detect faces first, then recognize only when needed
- Consider GPU acceleration

### Q: How to improve processing speed?
A:
1. Use OpenFace model (fastest)
2. Reduce image resolution
3. Adjust Pigo's ShiftFactor and ScaleFactor
4. Use batch processing with goroutines
5. Enable GPU acceleration in OpenCV (if available)

### Q: Different models give different results?
A: Yes, each model has different:
- Training data
- Architecture
- Feature dimensions
- Optimal thresholds

Test with your specific use case to find the best model.

## Model Comparison

### Speed Benchmark (on Intel i7, single core)
- OpenFace: ~15ms per face
- FaceNet: ~25ms per face
- ArcFace: ~30ms per face

### Accuracy (LFW dataset)
- OpenFace: ~92%
- FaceNet: ~99.63%
- ArcFace: ~99.82%

*Note: Actual performance varies based on hardware, image quality, and use case.*

## Advanced Usage

### Batch Processing with Progress Tracking

```go
type ProgressTracker struct {
    total     int
    processed int
    mu        sync.Mutex
}

func (pt *ProgressTracker) Update() {
    pt.mu.Lock()
    pt.processed++
    progress := float64(pt.processed) / float64(pt.total) * 100
    pt.mu.Unlock()
    fmt.Printf("\rProgress: %.1f%%", progress)
}

tracker := &ProgressTracker{total: len(images)}
// Use in concurrent processing...
```

### Custom Distance Metrics

```go
// The library uses cosine similarity by default
// You can implement custom metrics:

func customSimilarity(a, b []float32) float32 {
    // Your custom similarity calculation
    // Example: Euclidean distance
    var sum float32
    for i := range a {
        diff := a[i] - b[i]
        sum += diff * diff
    }
    return 1.0 / (1.0 + float32(math.Sqrt(float64(sum))))
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [Pigo](https://github.com/esimov/pigo) - Fast face detection
- [GoCV](https://gocv.io/) - Go bindings for OpenCV
- [OpenFace](https://cmusatyalab.github.io/openface/) - Face recognition toolkit
- [InsightFace](https://github.com/deepinsight/insightface) - State-of-the-art face recognition

## Acknowledgments

This library builds upon the excellent work of:
- Pigo face detector by Endre Simo
- GoCV by the Hybridgroup team
- OpenFace, FaceNet, and ArcFace research teams