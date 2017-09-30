import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TetrixComponent } from './tetrix.component';

describe('TetrixComponent', () => {
  let component: TetrixComponent;
  let fixture: ComponentFixture<TetrixComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TetrixComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(TetrixComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should be created', () => {
    expect(component).toBeTruthy();
  });
});
