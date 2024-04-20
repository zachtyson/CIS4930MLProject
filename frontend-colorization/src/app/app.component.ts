import { Component } from '@angular/core';
import { ImageColorizationService } from "./image-colorization.service";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  selectedFiles: File[] = [];
  successMessage: string = '';
  errorMessage: string = '';
  images: { original: string, processed?: string, expanded?: boolean }[] = [];

  toggleImage(index: number): void {
    this.images[index].expanded = !this.images[index].expanded;
  }

  removeImage(index: number): void {
    this.images.splice(index, 1); // Removes the image from the array
  }

  constructor(private fileUploadService: ImageColorizationService) { }

  onFileSelected(event: any) {
    const files: FileList = event.target.files;
    this.processFiles(files);
  }

  onFilesDropped(event: DragEvent) {
    event.preventDefault();
    if (event.dataTransfer && event.dataTransfer.files) {
      const files: FileList = event.dataTransfer.files;
      this.processFiles(files);
    }
  }

  onDragOver(event: Event) {
    event.preventDefault();
  }

  onDragLeave(event: Event) {
    event.preventDefault();
  }

  processFiles(files: FileList) {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.type.match('image/jpeg')) {
        const reader = new FileReader();
        reader.onload = (e: any) => {
          this.images.push({original: e.target.result});
        };
        reader.readAsDataURL(file);
        this.selectedFiles.push(file);
      } else {
        this.errorMessage = 'Only JPEG/JPG files are allowed.';
      }
    }
  }

  onUpload() {
    if (!this.selectedFiles.length) {
      this.errorMessage = 'Please select a file to upload.';
      return;
    }
    for (let j = 0; j < this.selectedFiles.length; j++) {
      this.fileUploadService.uploadFile(this.selectedFiles[j]).subscribe(response => {
        const i: ImageColorizationResponse = response as ImageColorizationResponse;
        this.successMessage = 'Upload success';
        this.errorMessage = ''; // Clear any previous error messages
        if (i.image) {
          this.images[j].processed = 'data:image/jpeg;base64,' + i.image;
        }
      }, error => {
        this.errorMessage = 'Upload error: ' + error.message;
        this.successMessage = ''; // Clear any previous success messages
      });
    }
  }
}

interface ImageColorizationResponse {
  image: string;
}

