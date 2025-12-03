package face

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"gocv.io/x/gocv"
)

// SupportedImageFormats lists all supported image formats
var SupportedImageFormats = []string{
	".jpg", ".jpeg", // JPEG
	".png",          // PNG
	".bmp",          // Bitmap
	".tif", ".tiff", // TIFF
	".webp", // WebP
	".gif",  // GIF
}

// IsSupportedImageFormat checks if the file extension is supported
func IsSupportedImageFormat(filename string) bool {
	ext := strings.ToLower(filepath.Ext(filename))
	for _, supportedExt := range SupportedImageFormats {
		if ext == supportedExt {
			return true
		}
	}
	return false
}

// LoadImage loads an image from file path
// Supports: JPG, PNG, BMP, TIFF, WebP, GIF
func LoadImage(filepath string) (gocv.Mat, error) {
	if !IsSupportedImageFormat(filepath) {
		return gocv.Mat{}, fmt.Errorf("unsupported image format: %s", filepath)
	}

	img := gocv.IMRead(filepath, gocv.IMReadColor)
	if img.Empty() {
		return gocv.Mat{}, fmt.Errorf("failed to load image: %s", filepath)
	}

	return img, nil
}

// LoadImageFromBytes loads an image from byte slice
func LoadImageFromBytes(data []byte) (gocv.Mat, error) {
	img, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to decode image: %v", err)
	}

	if img.Empty() {
		return gocv.Mat{}, fmt.Errorf("decoded image is empty")
	}

	return img, nil
}

// LoadImageFromStdImage converts standard Go image.Image to gocv.Mat
func LoadImageFromStdImage(img image.Image) (gocv.Mat, error) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	mat := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Convert from 16-bit to 8-bit
			mat.SetUCharAt(y, x*3+2, uint8(r>>8)) // R
			mat.SetUCharAt(y, x*3+1, uint8(g>>8)) // G
			mat.SetUCharAt(y, x*3, uint8(b>>8))   // B
		}
	}

	return mat, nil
}

// SaveImage saves a Mat to file
func SaveImage(filepath string, img gocv.Mat) error {
	if !IsSupportedImageFormat(filepath) {
		return fmt.Errorf("unsupported image format: %s", filepath)
	}

	success := gocv.IMWrite(filepath, img)
	if !success {
		return fmt.Errorf("failed to save image: %s", filepath)
	}

	return nil
}

// GetImageInfo returns information about an image file
func GetImageInfo(filepath string) (width, height, channels int, err error) {
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		return 0, 0, 0, fmt.Errorf("file does not exist: %s", filepath)
	}

	img := gocv.IMRead(filepath, gocv.IMReadColor)
	if img.Empty() {
		return 0, 0, 0, fmt.Errorf("failed to read image: %s", filepath)
	}
	defer img.Close()

	return img.Cols(), img.Rows(), img.Channels(), nil
}
