package face

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// FaceStorage defines the interface for face feature storage
// Implementations can use database, filesystem, or memory storage
type FaceStorage interface {
	// SavePerson saves a person and their features
	SavePerson(person *Person) error

	// LoadPerson loads a person by ID
	LoadPerson(id string) (*Person, error)

	// LoadAllPersons loads all persons
	LoadAllPersons() ([]*Person, error)

	// DeletePerson deletes a person by ID
	DeletePerson(id string) error

	// PersonExists checks if a person exists
	PersonExists(id string) (bool, error)

	// Close closes the storage connection
	Close() error
}

// MemoryStorage implements in-memory storage (default, fast but volatile)
type MemoryStorage struct {
	persons map[string]*Person
	mu      sync.RWMutex
}

// NewMemoryStorage creates a new in-memory storage
func NewMemoryStorage() *MemoryStorage {
	return &MemoryStorage{
		persons: make(map[string]*Person),
	}
}

func (s *MemoryStorage) SavePerson(person *Person) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Deep copy to avoid external modifications
	personCopy := &Person{
		ID:       person.ID,
		Name:     person.Name,
		Features: make([]FaceFeature, len(person.Features)),
	}
	copy(personCopy.Features, person.Features)

	s.persons[person.ID] = personCopy
	return nil
}

func (s *MemoryStorage) LoadPerson(id string) (*Person, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	person, exists := s.persons[id]
	if !exists {
		return nil, fmt.Errorf("person not found: %s", id)
	}

	// Return a copy
	personCopy := &Person{
		ID:       person.ID,
		Name:     person.Name,
		Features: make([]FaceFeature, len(person.Features)),
	}
	copy(personCopy.Features, person.Features)

	return personCopy, nil
}

func (s *MemoryStorage) LoadAllPersons() ([]*Person, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	persons := make([]*Person, 0, len(s.persons))
	for _, person := range s.persons {
		personCopy := &Person{
			ID:       person.ID,
			Name:     person.Name,
			Features: make([]FaceFeature, len(person.Features)),
		}
		copy(personCopy.Features, person.Features)
		persons = append(persons, personCopy)
	}

	return persons, nil
}

func (s *MemoryStorage) DeletePerson(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.persons[id]; !exists {
		return fmt.Errorf("person not found: %s", id)
	}

	delete(s.persons, id)
	return nil
}

func (s *MemoryStorage) PersonExists(id string) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	_, exists := s.persons[id]
	return exists, nil
}

func (s *MemoryStorage) Close() error {
	return nil
}

// FileStorage implements filesystem-based storage (persistent)
type FileStorage struct {
	baseDir string
	mu      sync.RWMutex
}

// NewFileStorage creates a new filesystem storage
func NewFileStorage(baseDir string) (*FileStorage, error) {
	// Create base directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %v", err)
	}

	return &FileStorage{
		baseDir: baseDir,
	}, nil
}

func (s *FileStorage) getPersonPath(id string) string {
	return filepath.Join(s.baseDir, fmt.Sprintf("%s.json", id))
}

func (s *FileStorage) SavePerson(person *Person) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := json.MarshalIndent(person, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal person: %v", err)
	}

	path := s.getPersonPath(person.ID)
	if err := ioutil.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write person file: %v", err)
	}

	return nil
}

func (s *FileStorage) LoadPerson(id string) (*Person, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	path := s.getPersonPath(id)
	data, err := ioutil.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("person not found: %s", id)
		}
		return nil, fmt.Errorf("failed to read person file: %v", err)
	}

	var person Person
	if err := json.Unmarshal(data, &person); err != nil {
		return nil, fmt.Errorf("failed to unmarshal person: %v", err)
	}

	return &person, nil
}

func (s *FileStorage) LoadAllPersons() ([]*Person, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	files, err := ioutil.ReadDir(s.baseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read storage directory: %v", err)
	}

	persons := make([]*Person, 0)
	for _, file := range files {
		if file.IsDir() || filepath.Ext(file.Name()) != ".json" {
			continue
		}

		id := file.Name()[:len(file.Name())-5] // Remove .json extension
		person, err := s.LoadPerson(id)
		if err != nil {
			// Skip corrupted files
			continue
		}

		persons = append(persons, person)
	}

	return persons, nil
}

func (s *FileStorage) DeletePerson(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	path := s.getPersonPath(id)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("person not found: %s", id)
	}

	if err := os.Remove(path); err != nil {
		return fmt.Errorf("failed to delete person file: %v", err)
	}

	return nil
}

func (s *FileStorage) PersonExists(id string) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	path := s.getPersonPath(id)
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func (s *FileStorage) Close() error {
	return nil
}

// JSONStorage implements a single JSON file storage (for small datasets)
type JSONStorage struct {
	filepath string
	persons  map[string]*Person
	mu       sync.RWMutex
}

// NewJSONStorage creates a new JSON file storage
func NewJSONStorage(filepath string) (*JSONStorage, error) {
	storage := &JSONStorage{
		filepath: filepath,
		persons:  make(map[string]*Person),
	}

	// Try to load existing data
	if _, err := os.Stat(filepath); err == nil {
		if err := storage.load(); err != nil {
			return nil, fmt.Errorf("failed to load existing data: %v", err)
		}
	}

	return storage, nil
}

func (s *JSONStorage) load() error {
	data, err := ioutil.ReadFile(s.filepath)
	if err != nil {
		return err
	}

	return json.Unmarshal(data, &s.persons)
}

func (s *JSONStorage) save() error {
	data, err := json.MarshalIndent(s.persons, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(s.filepath, data, 0644)
}

func (s *JSONStorage) SavePerson(person *Person) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.persons[person.ID] = person
	return s.save()
}

func (s *JSONStorage) LoadPerson(id string) (*Person, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	person, exists := s.persons[id]
	if !exists {
		return nil, fmt.Errorf("person not found: %s", id)
	}

	return person, nil
}

func (s *JSONStorage) LoadAllPersons() ([]*Person, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	persons := make([]*Person, 0, len(s.persons))
	for _, person := range s.persons {
		persons = append(persons, person)
	}

	return persons, nil
}

func (s *JSONStorage) DeletePerson(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.persons[id]; !exists {
		return fmt.Errorf("person not found: %s", id)
	}

	delete(s.persons, id)
	return s.save()
}

func (s *JSONStorage) PersonExists(id string) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	_, exists := s.persons[id]
	return exists, nil
}

func (s *JSONStorage) Close() error {
	return s.save()
}

// StorageMetadata contains metadata about stored persons
type StorageMetadata struct {
	TotalPersons  int       `json:"total_persons"`
	TotalFeatures int       `json:"total_features"`
	LastUpdated   time.Time `json:"last_updated"`
}

// GetMetadata returns metadata about the storage
func GetStorageMetadata(storage FaceStorage) (*StorageMetadata, error) {
	persons, err := storage.LoadAllPersons()
	if err != nil {
		return nil, err
	}

	totalFeatures := 0
	for _, person := range persons {
		totalFeatures += len(person.Features)
	}

	return &StorageMetadata{
		TotalPersons:  len(persons),
		TotalFeatures: totalFeatures,
		LastUpdated:   time.Now(),
	}, nil
}
