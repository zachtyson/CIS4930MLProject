import { Component, ElementRef, ViewChild } from '@angular/core';
import { ImageColorizationService } from "./image-colorization.service";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  @ViewChild('fileInput') fileInput: ElementRef<HTMLInputElement> | undefined;
  successMessage: string = '';
  errorMessage: string = '';
  images: { original: string, processed?: string, expanded?: boolean }[] = [];

  constructor(private fileUploadService: ImageColorizationService) { }

  addImages(): void {
    if (!this.fileInput) {
      return;
    }
    this.fileInput.nativeElement.click();
  }

  toggleImage(index: number): void {
    this.images[index].expanded = !this.images[index].expanded;
  }

  removeImage(index: number): void {
    this.images.splice(index, 1); // Removes the image from the array
  }

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
          const imgData = { original: e.target.result, expanded: false };
          this.images.push(imgData);
          this.uploadImage(file, this.images.length - 1);
        };
        reader.readAsDataURL(file);
      } else {
        this.errorMessage = 'Only JPEG/JPG files are allowed.';
      }
    }
  }

  uploadImage(file: File, index: number) {
    this.fileUploadService.uploadFile(file).subscribe(response => {
      const imgResponse: ImageColorizationResponse = response as ImageColorizationResponse;
      this.successMessage = 'Upload success';
      this.errorMessage = '';
      if (imgResponse.image) {
        this.images[index].processed = 'data:image/jpeg;base64,' + imgResponse.image;
      }
    }, error => {
      this.errorMessage = 'Upload error: ' + error.message;
      this.successMessage = '';
    });
  }
}

interface ImageColorizationResponse {
  image: string;
}
