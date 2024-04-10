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
  outputImgSrcs: string[] | null = null;
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
    this.outputImgSrcs = [];
    this.successMessage = '';
    this.originalImageSrc = '';

  }

  onUpload() {
    if (!this.selectedFile) {
      this.errorMessage = 'Please select a file to upload.';
      return;
    }
    this.fileUploadService.uploadFile(this.selectedFile).subscribe(response => {
      console.log(response);
      const res: any = response as any;
      this.successMessage = 'Upload success';
      this.errorMessage = ''; // Clear any previous error messages
      // check for keys 'colorization_model_combo_' + i
      this.outputImgSrcs = [];
      for (let i = 0; i < 126; i++) {
        const key = 'colorization_model_combo_' + i+ '.pth';
        if (key in res) {
          // @ts-ignore
          this.outputImgSrcs.push('data:image/jpeg;base64,' + res[key]);
        }
      }
      console.log(this.outputImgSrcs);
    }, error => {
      this.errorMessage = 'Upload error: ' + error.message;
      this.successMessage = ''; // Clear any previous success messages
      this.outputImgSrcs = null;
    });
  }
}

interface ImageColorizationResponse {
  image: string;
}
