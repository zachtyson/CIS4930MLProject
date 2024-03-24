import { Component } from '@angular/core';
import { ImageColorizationService } from "./image-colorization.service";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  selectedFile: File | null = null;
  originalImageSrc: string = '';
  imageSrc: string = '';
  successMessage: string = '';
  errorMessage: string = '';

  constructor(private fileUploadService: ImageColorizationService) { }

  onFileSelected(event: any) {
    const file: File = event.target.files[0];

    if (file) {
      // Check if the file is a JPEG/JPG.
      if (file.type.match('image/jpeg')) {
        this.selectedFile = file;
        // Convert original image to base64.
        const reader = new FileReader();
        reader.onload = (e: any) => {
          this.originalImageSrc = e.target.result; // The Base64 string.
        };
        reader.readAsDataURL(this.selectedFile);
        this.errorMessage = '';
      } else {
        this.errorMessage = 'Only JPEG/JPG files are allowed.';
        this.selectedFile = null;
      }
    } else {
      this.errorMessage = 'Please select a file.';
      this.selectedFile = null;
      this.successMessage = '';
    }
    this.imageSrc = '';
    this.successMessage = '';
    this.originalImageSrc = '';

  }

  onUpload() {
    if (!this.selectedFile) {
      this.errorMessage = 'Please select a file to upload.';
      return;
    }
    this.fileUploadService.uploadFile(this.selectedFile).subscribe(response => {
      const i: ImageColorizationResponse = response as ImageColorizationResponse;
      this.successMessage = 'Upload success';
      this.errorMessage = ''; // Clear any previous error messages
      if (i.image) {
        this.imageSrc = 'data:image/jpeg;base64,' + i.image;
      }
    }, error => {
      this.errorMessage = 'Upload error: ' + error.message;
      this.successMessage = ''; // Clear any previous success messages
    });
  }
}

interface ImageColorizationResponse {
  image: string;
}
