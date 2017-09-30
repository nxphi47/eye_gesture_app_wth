import { TestBed, inject } from '@angular/core/testing';

import { InferenceService } from './inference.service';

describe('InferenceService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [InferenceService]
    });
  });

  it('should be created', inject([InferenceService], (service: InferenceService) => {
    expect(service).toBeTruthy();
  }));
});
