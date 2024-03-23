import { TestBed } from '@angular/core/testing';

import { ImageColorizationService } from './image-colorization.service';

describe('ImageColorizationService', () => {
  let service: ImageColorizationService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ImageColorizationService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
