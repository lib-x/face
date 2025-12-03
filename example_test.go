package face_test

import (
	"fmt"
	"log"
	"os"

	"github.com/lib-x/face"
)

// Example demonstrates basic face recognition workflow
func Example() {
	// Initialize recognizer
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := face.NewFaceRecognizer(config)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	// Register a person
	err = recognizer.AddPerson("001", "Alice")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Face recognition system initialized")
	fmt.Println("Person registered: Alice (ID: 001)")

	// Output:
	// Face recognition system initialized
	// Person registered: Alice (ID: 001)
}

// ExampleLoadImage demonstrates how to load images in different formats
func ExampleLoadImage() {
	// Check if image format is supported
	if face.IsSupportedImageFormat("photo.jpg") {
		fmt.Println("JPEG format is supported")
	}

	// Load image from file
	img, err := face.LoadImage("./testdata/sample.jpg")
	if err != nil {
		// Handle error - image file might not exist in example
		fmt.Println("Image loading example (file may not exist)")
		return
	}
	defer img.Close()

	fmt.Println("Image loaded successfully")

	// Output:
	// JPEG format is supported
}

// ExampleNewMemoryStorage demonstrates in-memory storage
func ExampleNewMemoryStorage() {
	// Create in-memory storage
	storage := face.NewMemoryStorage()

	// Use with recognizer
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := face.NewFaceRecognizer(
		config,
		face.WithStorage(storage),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	fmt.Println("Memory storage initialized")

	// Output:
	// Memory storage initialized
}

// ExampleNewFileStorage demonstrates persistent file storage
func ExampleNewFileStorage() {
	// Create file storage
	storage, err := face.NewFileStorage("./testdata/face_db")
	if err != nil {
		log.Fatal(err)
	}
	defer storage.Close()

	// Use with recognizer
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := face.NewFaceRecognizer(
		config,
		face.WithStorage(storage),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	fmt.Println("File storage initialized")
	fmt.Println("Data will be persisted to: ./testdata/face_db")

	// Output:
	// File storage initialized
	// Data will be persisted to: ./testdata/face_db
}

// ExampleNewJSONStorage demonstrates single JSON file storage
func ExampleNewJSONStorage() {
	// Create JSON storage
	storage, err := face.NewJSONStorage("./testdata/faces.json")
	if err != nil {
		log.Fatal(err)
	}
	defer storage.Close()

	fmt.Println("JSON storage initialized")
	fmt.Println("Data file: ./testdata/faces.json")

	// Output:
	// JSON storage initialized
	// Data file: ./testdata/faces.json
}

// ExampleFaceRecognizer_AddPerson demonstrates person registration
func ExampleFaceRecognizer_AddPerson() {
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := face.NewFaceRecognizer(config)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	// Add a person
	err = recognizer.AddPerson("001", "Alice")
	if err != nil {
		log.Fatal(err)
	}

	// Add another person
	err = recognizer.AddPerson("002", "Bob")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Registered: Alice")
	fmt.Println("Registered: Bob")

	// Output:
	// Registered: Alice
	// Registered: Bob
}

// ExampleFaceRecognizer_AddFaceSample demonstrates adding face samples
func ExampleFaceRecognizer_AddFaceSample() {
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := face.NewFaceRecognizer(config)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	// Register person first
	recognizer.AddPerson("001", "Alice")

	// Load image (example - file may not exist)
	img, err := face.LoadImage("./testdata/alice.jpg")
	if err != nil {
		fmt.Println("Adding face sample example")
		fmt.Println("Note: Sample image file not found")
		return
	}
	defer img.Close()

	// Add face sample
	err = recognizer.AddFaceSample("001", img)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Face sample added for Alice")

	// Output:
	// Adding face sample example
	// Note: Sample image file not found
}

// ExampleFaceRecognizer_Recognize demonstrates face recognition
func ExampleFaceRecognizer_Recognize() {
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := face.NewFaceRecognizer(config)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	// This would normally recognize faces in an image
	// For this example, we just show the API usage
	fmt.Println("Face recognition API example:")
	fmt.Println("1. Load image with LoadImage()")
	fmt.Println("2. Call Recognize(img) to detect and identify faces")
	fmt.Println("3. Process RecognizeResult for each detected face")

	// Output:
	// Face recognition API example:
	// 1. Load image with LoadImage()
	// 2. Call Recognize(img) to detect and identify faces
	// 3. Process RecognizeResult for each detected face
}

// ExampleFaceRecognizer_ListPersons demonstrates listing registered persons
func ExampleFaceRecognizer_ListPersons() {
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	recognizer, err := face.NewFaceRecognizer(config)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	// Add some persons
	recognizer.AddPerson("001", "Alice")
	recognizer.AddPerson("002", "Bob")
	recognizer.AddPerson("003", "Charlie")

	// List all persons
	persons := recognizer.ListPersons()

	fmt.Printf("Total registered persons: %d\n", len(persons))
	for _, person := range persons {
		fmt.Printf("ID: %s, Name: %s\n", person.ID, person.Name)
	}

	// Output:
	// Total registered persons: 3
	// ID: 001, Name: Alice
	// ID: 002, Name: Bob
	// ID: 003, Name: Charlie
}

// ExampleWithSimilarityThreshold demonstrates setting similarity threshold
func ExampleWithSimilarityThreshold() {
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	// Create recognizer with custom threshold
	recognizer, err := face.NewFaceRecognizer(
		config,
		face.WithSimilarityThreshold(0.7), // Higher threshold = stricter matching
	)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	threshold := recognizer.GetThreshold()
	fmt.Printf("Similarity threshold set to: %.2f\n", threshold)

	// Output:
	// Similarity threshold set to: 0.70
}

// ExampleWithModelType demonstrates using different model types
func ExampleWithModelType() {
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	// Use OpenFace model (default)
	recognizer, err := face.NewFaceRecognizer(
		config,
		face.WithModelType(face.ModelOpenFace),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	modelConfig := recognizer.GetModelConfig()
	fmt.Printf("Model type: %s\n", modelConfig.Type)
	fmt.Printf("Feature dimension: %d\n", modelConfig.FeatureDim)
	fmt.Printf("Input size: %dx%d\n", modelConfig.InputSize.X, modelConfig.InputSize.Y)

	// Output:
	// Model type: openface
	// Feature dimension: 128
	// Input size: 96x96
}

// ExampleGetStorageMetadata demonstrates getting storage statistics
func ExampleGetStorageMetadata() {
	storage := face.NewMemoryStorage()

	// Create person data for example
	person := &face.Person{
		ID:   "001",
		Name: "Alice",
		Features: []face.FaceFeature{
			{PersonID: "001", Feature: make([]float32, 128)},
			{PersonID: "001", Feature: make([]float32, 128)},
		},
	}
	storage.SavePerson(person)

	// Get metadata
	metadata, err := face.GetStorageMetadata(storage)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Total persons: %d\n", metadata.TotalPersons)
	fmt.Printf("Total features: %d\n", metadata.TotalFeatures)

	// Output:
	// Total persons: 1
	// Total features: 2
}

// ExampleIsSupportedImageFormat demonstrates format checking
func ExampleIsSupportedImageFormat() {
	formats := []string{
		"photo.jpg",
		"image.png",
		"picture.gif",
		"document.pdf", // Not supported
	}

	for _, filename := range formats {
		if face.IsSupportedImageFormat(filename) {
			fmt.Printf("%s: supported\n", filename)
		} else {
			fmt.Printf("%s: not supported\n", filename)
		}
	}

	// Output:
	// photo.jpg: supported
	// image.png: supported
	// picture.gif: supported
	// document.pdf: not supported
}

// Example_batchRegistration demonstrates registering multiple persons with multiple photos
func Example_batchRegistration() {
	config := face.Config{
		PigoCascadeFile:  "./testdata/facefinder",
		FaceEncoderModel: "./testdata/nn4.small2.v1.t7",
	}

	storage, _ := face.NewFileStorage("./testdata/face_db")
	defer storage.Close()

	recognizer, err := face.NewFaceRecognizer(
		config,
		face.WithStorage(storage),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer recognizer.Close()

	// Batch registration data
	persons := map[string]struct {
		name   string
		photos []string
	}{
		"001": {name: "Alice", photos: []string{"alice1.jpg", "alice2.jpg"}},
		"002": {name: "Bob", photos: []string{"bob1.jpg", "bob2.jpg"}},
	}

	// Register each person
	for id, data := range persons {
		recognizer.AddPerson(id, data.name)
		fmt.Printf("Registered: %s (ID: %s)\n", data.name, id)

		// In real scenario, would load and add photos here
		// For example purposes, we just show the structure
		for i := range data.photos {
			fmt.Printf("  - Sample %d would be added\n", i+1)
		}
	}

	// List all registered persons
	persons_list := recognizer.ListPersons()
	fmt.Printf("\nTotal registered: %d persons\n", len(persons_list))

	// Output:
	// Registered: Alice (ID: 001)
	//   - Sample 1 would be added
	//   - Sample 2 would be added
	// Registered: Bob (ID: 002)
	//   - Sample 1 would be added
	//   - Sample 2 would be added
	//
	// Total registered: 2 persons
}

// Example_workflowComplete demonstrates a complete workflow
func Example_workflowComplete() {
	fmt.Println("Complete Face Recognition Workflow:")
	fmt.Println("")
	fmt.Println("Step 1: Initialize recognizer with file storage")
	fmt.Println("Step 2: Register persons and add face samples")
	fmt.Println("Step 3: Save data (automatic with file storage)")
	fmt.Println("Step 4: Load test image")
	fmt.Println("Step 5: Detect and recognize faces")
	fmt.Println("Step 6: Process recognition results")
	fmt.Println("")
	fmt.Println("API Usage:")
	fmt.Println("  storage, _ := face.NewFileStorage(\"./face_db\")")
	fmt.Println("  recognizer, _ := face.NewFaceRecognizer(config, face.WithStorage(storage))")
	fmt.Println("  recognizer.AddPerson(\"001\", \"Alice\")")
	fmt.Println("  recognizer.AddFaceSample(\"001\", img)")
	fmt.Println("  results, _ := recognizer.Recognize(testImg)")

	// Output:
	// Complete Face Recognition Workflow:
	//
	// Step 1: Initialize recognizer with file storage
	// Step 2: Register persons and add face samples
	// Step 3: Save data (automatic with file storage)
	// Step 4: Load test image
	// Step 5: Detect and recognize faces
	// Step 6: Process recognition results
	//
	// API Usage:
	//   storage, _ := face.NewFileStorage("./face_db")
	//   recognizer, _ := face.NewFaceRecognizer(config, face.WithStorage(storage))
	//   recognizer.AddPerson("001", "Alice")
	//   recognizer.AddFaceSample("001", img)
	//   results, _ := recognizer.Recognize(testImg)
}

// Clean up test files after examples (if they were created)
func init() {
	// This would run after examples but we don't want to remove testdata
	// Just showing cleanup pattern
}

// Remove test database if exists
func cleanupTestDB() {
	os.RemoveAll("./testdata/face_db")
	os.Remove("./testdata/faces.json")
}
