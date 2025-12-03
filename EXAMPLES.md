# Face Recognition Library Examples

This document explains how to run and view the examples for the face recognition library.

## Prerequisites

1. **Download models first** (required):
```bash
cd cmd/download_test
go run main.go --model required --proxy socks5://127.0.0.1:10808
```

This will download:
- `facefinder` (235KB) - Pigo face detector
- `nn4.small2.v1.t7` (30MB) - OpenFace face recognition model

Models will be saved to `./models/` directory.

2. **Copy models to testdata** (for examples):
```bash
mkdir -p testdata
cp models/facefinder testdata/
cp models/nn4.small2.v1.t7 testdata/
```

## Viewing Examples

The library includes Go-style examples that can be viewed with `go doc` or run with `go test`.

### List all examples:
```bash
go doc -all | grep "Example"
```

### View specific example:
```bash
go doc Example
go doc ExampleNewFileStorage
go doc ExampleFaceRecognizer_AddPerson
```

### Run all examples:
```bash
go test -v -run Example
```

### Run specific example:
```bash
go test -v -run ExampleNewFileStorage
```

## Available Examples

### Basic Examples

1. **Example** - Basic face recognition workflow
   ```bash
   go doc Example
   ```

2. **ExampleLoadImage** - Loading images in different formats
   ```bash
   go doc ExampleLoadImage
   ```

3. **ExampleIsSupportedImageFormat** - Checking supported formats
   ```bash
   go doc ExampleIsSupportedImageFormat
   ```

### Storage Examples

4. **ExampleNewMemoryStorage** - In-memory storage (fast, volatile)
   ```bash
   go doc ExampleNewMemoryStorage
   ```

5. **ExampleNewFileStorage** - File storage (persistent)
   ```bash
   go doc ExampleNewFileStorage
   ```

6. **ExampleNewJSONStorage** - Single JSON file storage
   ```bash
   go doc ExampleNewJSONStorage
   ```

7. **ExampleGetStorageMetadata** - Storage statistics
   ```bash
   go doc ExampleGetStorageMetadata
   ```

### Person Management Examples

8. **ExampleFaceRecognizer_AddPerson** - Register persons
   ```bash
   go doc ExampleFaceRecognizer_AddPerson
   ```

9. **ExampleFaceRecognizer_AddFaceSample** - Add face samples
   ```bash
   go doc ExampleFaceRecognizer_AddFaceSample
   ```

10. **ExampleFaceRecognizer_ListPersons** - List registered persons
    ```bash
    go doc ExampleFaceRecognizer_ListPersons
    ```

### Recognition Examples

11. **ExampleFaceRecognizer_Recognize** - Face recognition
    ```bash
    go doc ExampleFaceRecognizer_Recognize
    ```

12. **ExampleWithSimilarityThreshold** - Set recognition threshold
    ```bash
    go doc ExampleWithSimilarityThreshold
    ```

13. **ExampleWithModelType** - Use different model types
    ```bash
    go doc ExampleWithModelType
    ```

### Advanced Examples

14. **Example_batchRegistration** - Batch registration
    ```bash
    go doc Example_batchRegistration
    ```

15. **Example_workflowComplete** - Complete workflow
    ```bash
    go doc Example_workflowComplete
    ```

## Example Code Patterns

### Pattern 1: Basic Setup
```go
config := face.Config{
    PigoCascadeFile:  "./testdata/facefinder",
    FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
}

recognizer, err := face.NewFaceRecognizer(config)
if err != nil {
    log.Fatal(err)
}
defer recognizer.Close()
```

### Pattern 2: With File Storage
```go
storage, _ := face.NewFileStorage("./face_database")
defer storage.Close()

recognizer, _ := face.NewFaceRecognizer(
    config,
    face.WithStorage(storage),
    face.WithSimilarityThreshold(0.6),
)
defer recognizer.Close()
```

### Pattern 3: Register Person
```go
// Add person
recognizer.AddPerson("001", "Alice")

// Load and add face sample
img, _ := face.LoadImage("alice.jpg")
defer img.Close()

recognizer.AddFaceSample("001", img)
```

### Pattern 4: Recognize Faces
```go
// Load test image
img, _ := face.LoadImage("test.jpg")
defer img.Close()

// Recognize
results, _ := recognizer.Recognize(img)

// Process results
for _, result := range results {
    fmt.Printf("%s: %.2f%%\n",
        result.PersonName,
        result.Confidence*100)
}
```

## Testing Examples

Run examples with different options:

```bash
# Run all examples
go test -v -run Example

# Run only storage examples
go test -v -run "Example.*Storage"

# Run only recognizer examples
go test -v -run "ExampleFaceRecognizer"

# Show example output
go test -v -run ExampleWithModelType
```

## Supported Image Formats

The library supports:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- WebP (.webp)
- GIF (.gif)

## Storage Options

Three storage backends are available:

1. **Memory Storage** - Fast, volatile (default)
   - Data lost on restart
   - Best for: Testing, temporary use

2. **File Storage** - Persistent, per-person files
   - Each person stored in separate JSON file
   - Best for: Production, medium-scale apps

3. **JSON Storage** - Persistent, single file
   - All data in one JSON file
   - Best for: Small-scale apps, simple deployments

## Common Issues

### 1. Models not found
```
Error: failed to read Pigo cascade file: no such file or directory
```
**Solution**: Download models and copy to testdata/
```bash
cd cmd/download_test
go run main.go --model required --proxy socks5://127.0.0.1:10808
mkdir -p ../../testdata
cp models/* ../../testdata/
```

### 2. Image file not found
Most examples will handle missing image files gracefully by showing the API usage pattern instead of failing.

### 3. OpenCV not installed
```
Error: package gocv.io/x/gocv: C source files not allowed
```
**Solution**: Install OpenCV and gocv dependencies
```bash
# macOS
brew install opencv

# Ubuntu/Debian
sudo apt-get install libopencv-dev

# Then install gocv
go get -u -d gocv.io/x/gocv
cd $GOPATH/src/gocv.io/x/gocv
make install
```

## Running the API Server

For a complete HTTP API example:

```bash
cd cmd/api_server
go run main.go
```

Then test with curl:
```bash
# Register person
curl -X POST http://localhost:8080/api/register \
  -F "person_id=001" \
  -F "person_name=Alice" \
  -F "images=@photo.jpg"

# Recognize face
curl -X POST http://localhost:8080/api/recognize \
  -F "image=@test.jpg"

# List persons
curl http://localhost:8080/api/persons
```

## Next Steps

1. Review `USAGE_GUIDE.md` for detailed API documentation
2. Check `example_test.go` for example source code
3. See `storage.go` for custom storage implementation
4. Read `README.md` for library overview

## Getting Help

- View inline documentation: `go doc github.com/lib-x/face`
- Run examples: `go test -v -run Example`
- Check test coverage: `go test -cover`
- Report issues: Create an issue in the repository
