import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { EyeIdComponent } from './eye-id.component';

describe('EyeIdComponent', () => {
  let component: EyeIdComponent;
  let fixture: ComponentFixture<EyeIdComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ EyeIdComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(EyeIdComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should be created', () => {
    expect(component).toBeTruthy();
  });
});
