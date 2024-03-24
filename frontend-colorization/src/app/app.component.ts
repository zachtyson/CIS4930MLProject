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

  onFileSelected(event:any) {
    this.selectedFile = <File>event.target.files[0];
    // convert original image to base64
    if(this.selectedFile) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.originalImageSrc = e.target.result;
      }
      reader.readAsDataURL(this.selectedFile);
    }
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
