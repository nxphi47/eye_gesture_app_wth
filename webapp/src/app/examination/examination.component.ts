import {Component, Input, OnInit} from '@angular/core';
import {Question} from "./question";
import {InferenceService} from "../services/inference.service";

@Component({
  selector: 'app-examination',
  templateUrl: './examination.component.html',
  styleUrls: ['./examination.component.css']
})
export class ExaminationComponent implements OnInit {

  @Input()
  ip: string;
  
  mappingLabels: string[] = ['left', 'right', 'up', 'down'];
  
  examining: boolean = false;
  
  time: number = 0;
  
  questions: Question[];
  
  focusQuestion: Question;
  focusChoice: number;
  currentLabel: string;
  
  headsetStatus: string = 'off';
  
  result: number;
  constructor(
    private inferenceService: InferenceService,
  ) {
    this.initQuestions();
    this.result = null;
  }

  ngOnInit() {
  }
  
  initQuestions(){
    this.questions = [
      new Question(`When was Singapore Born?`, ['1960', '1965', '1970'], 1),
      new Question(`How many countries in South East Asia?`, ['11', '10', '9'], 0),
      new Question(`Who invented the theory of Relativity?`, ['Max Plank', 'Thomas Edition', 'Albert Einstein'], 2),
      new Question(`1 + 1 = ?`, ['1', '2', '3'], 1),
    ];
  }
  
  selectChoice(label: string, question: Question) {
    question.select(this.mappingLabels.indexOf(label));
  }
  
  startExam() {
    this.reset();
    // start the time
    this.time = 10 * 60;
    
    this.examining = true;
    this.focusQuestion = this.questions[0];
    this.focusChoice = 0;
    // start the loop
    this.startClock();
    this.startEyeLoop();
  }
  
  get minutesTime() {
    return `${Math.floor(this.time / 60)} : ${this.time - 60 * Math.floor(this.time / 60)}`;
  }
  
  
  startClock() {
    if (!this.examining) return;
    this.time--;
    if (this.time > 0) {
      setTimeout(() => {
        
        this.startClock()
      }, 1000);
    }
    else {
      this.time = 0;
      alert('Time out');
      // return end result or yours
    }
  }
  
  startEyeLoop() {
    if (!this.examining && this.time > 0) {
      // finish
      this.headsetStatus = 'off';
      return;
    }
    this.headsetStatus = 'scanning';
    this.inferenceService.inference(this.ip)
      .subscribe(
        response => {
          console.log(response);
          this.currentLabel = response.label;
          if (this.validateLabel(response.label)) {
            if (response.label == 'double_blink' && this.focusQuestion.selectedChoice != null) {
              // toggle next question
              let currentInd = this.questions.indexOf(this.focusQuestion);
              if (currentInd == this.questions.length - 1) {
                // sure to submit ??? and submit
                this.endTest();
                this.focusQuestion = null;
              }
              else {
                this.focusQuestion = this.questions[++currentInd];
              }
            }
            else {
              this.selectChoice(response.label, this.focusQuestion);
            }
          }
          this.headsetStatus = 'waiting';
          setTimeout(() => {
            this.startEyeLoop();
          }, 1000);
        },
        error => {
          alert('Failed to connect to headset');
          this.reset();
        }
      );
    
  }
  
  endTest() {
    this.examining = false;
    this.result = 0;
    this.questions.forEach(v => {
      if (v.validate()) {
        this.result++;
      }
    });
    this.focusQuestion = null;
  }
  
  reset() {
    this.examining = false;
    this.result= null;
    this.headsetStatus = 'off';
    this.focusChoice = null;
    this.focusQuestion = null;
    this.time = 0;
  }
  
  validateLabel(label) {
    return (this.mappingLabels.includes(label) || label == 'double_blink');
  }
  

}
