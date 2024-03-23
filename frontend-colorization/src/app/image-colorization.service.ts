import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class ImageColorizationService {

  constructor(private http: HttpClient) {
  }

  uploadFile(file: File) {
    const formData = new FormData();
    formData.append('file', file, file.name);

    return this.http.post('http://localhost:8000/api/colorize', formData, {
      headers: new HttpHeaders({
        'Accept': 'application/json'
      }),
      responseType: 'json'
    });
  }


}
