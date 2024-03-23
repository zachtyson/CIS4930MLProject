// app.component.ts
import { Component } from '@angular/core';
import { ImageColorizationService } from "./image-colorization.service";


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  selectedFile: File | null = null;
  imageSrc: string = ''; // Add this line

  constructor(private fileUploadService: ImageColorizationService) { }

  onFileSelected(event:any) {
    this.selectedFile = <File>event.target.files[0];
  }

  onUpload() {
    if (!this.selectedFile) {
      return;
    }
    this.fileUploadService.uploadFile(this.selectedFile).subscribe(response => {
      const i: ImageColorizationResponse = response as ImageColorizationResponse;
      console.log('Upload success', i);
      if (i.image) { // Assuming 'image' is the key in the response containing the base64 string
        this.imageSrc = 'data:image/jpeg;base64,' + i.image;
      }
    }, error => {
      console.error('Upload error', error);
    });
  }
}

interface ImageColorizationResponse {
  image: string;
}
