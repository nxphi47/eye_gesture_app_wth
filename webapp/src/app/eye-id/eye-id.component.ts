import {Component, Input, OnInit} from '@angular/core';
import {validate} from "codelyzer/walkerFactory/walkerFn";
import {InferenceResponse, InferenceService} from "../services/inference.service";
declare var $: any;

@Component({
  selector: 'app-eye-id',
  templateUrl: './eye-id.component.html',
  styleUrls: ['./eye-id.component.css']
})
export class EyeIdComponent implements OnInit {
  
  labelSet = ['left', 'right', 'up', 'down'];
  inputs: string[];
  inputStates: boolean[];
  
  passcode = ['right', 'up', 'right', 'left'];
  
  length = 4;
  status: string = "";
  isUnlocking: boolean = false;
  requesting: boolean = false;
  
  @Input()
  ip: string;
  
  constructor(private inferenceService: InferenceService,) {
    this.inputs = [];
    this.inputStates = [false, false, false, false];
  }
  
  ngOnInit() {
  }
  
  // passcodeTransform(gestureArray: string[]): number[] {
  //   return gestureArray.map(v => this.labelSet.indexOf(v)).join("");
  // }
  
  validate(gestureArray: string[]): boolean {
    if (gestureArray.length != this.passcode.length) return false;
    for (let i = 0; i < gestureArray.length && i < this.passcode.length; i++) {
      if (gestureArray[i] != this.passcode[i]) return false;
    }
    return true;
  }
  
  resetWrong() {
    this.status = 'wrong';
    setTimeout(() => {
      this.reset();
    }, 500);
  }
  
  resetCorrect() {
    this.status = 'correct';
    setTimeout(() => {
      this.reset();
    }, 500);
  }
  
  reset() {
    this.inputs = [];
    this.inputStates = [false, false, false, false];
    this.isUnlocking = false;
    this.status = '';
  }
  
  start() {
    // let test = ['right', 'right', 'right', 'right'];
    let test = ['right', 'up', 'right', 'left'];
    //   setTimeout(()  => {
    //     this.appendCode(v)
    //   },1000);
    // })
    let i = 0;
    this.recurStart(test, 0);
  }
  
  enter() {
    this.requesting = !this.requesting;
    if (this.requesting) {
      this.startCodeEnquire();
    }
  }
  
  startCodeEnquire() {
    if (!this.requesting) {
      return;
    }
    // let response = this.inferenceService.inference();
    // console.log(response);
    this.inferenceService.inference(this.ip)
      .subscribe(
        response => {
          console.log(response);
          if (this.validateResponse(response)) {
            if (response.label == 'double_blink') {
              // toggle unlocking
              this.isUnlocking = !this.isUnlocking;
              if (!this.isUnlocking) {
                this.reset();
              }
            }
            else {
              if (this.isUnlocking) {
                this.appendCode(response.label);
              }
              else {
                this.reset();
              }
            }
          }
          setTimeout(() => {
            this.startCodeEnquire();
          }, 1500);
        }
      );
  }
  
  validateResponse(res: InferenceResponse) {
    if (!res) return false;
    // double
    if (res.label == 'double_blink') {
      // toggle unlocking
      // this.isUnlocking = !this.isUnlocking;
      return true;
    }
    if (!this.labelSet.includes(res.label)) return false;
    else return true;
  }
  
  recurStart(test, i) {
    if (i >= test.length) return;
    this.appendCode(test[i]);
    setTimeout(() => {
      this.recurStart(test, i + 1);
    }, 800)
  }
  
  appendCode(label: string) {
    if (this.inputs.length >= this.length) return;
    this.inputs.push(label);
    this.inputStates[this.inputs.length - 1] = true;
    
    if (this.inputs.length == this.passcode.length) {
      setTimeout(() => {
        if (this.validate(this.inputs)) {
          // if true
          this.resetCorrect();
          // alert('you are  correct');
        }
        else {
          // notify incorrect
          // alert(`Passcode in correct`);
          this.resetWrong();
          // alert('You wrong');
        }
      }, 100)
    }
  }
  
  render() {
  
  }
  
  init() {
    var input = '';
    
    
    var dots = document.getElementsByClassName('dot'),
      numbers = document.getElementsByClassName('number');
  }
  
}
