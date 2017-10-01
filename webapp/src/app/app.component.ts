import {Component} from '@angular/core';
import {InferenceResponse, InferenceService} from "./services/inference.service";

@Component({
	selector: 'app-root',
	templateUrl: './app.component.html',
	styleUrls: ['./app.component.css']
})
export class AppComponent {
	title = 'app';
	app: string = 'eyeid';
	ipAddress: string = "192.168.43.165";
	// headset 192.168.43.165
  // PI: .15
	connected: boolean = false;
	connecting: boolean = false;
	interval: number = 10;
	response: InferenceResponse;
	constructor(private inferenceService: InferenceService) {

	}
	
  connect() {
	  this.connecting = true;
	  this.inferenceService.connect(this.ipAddress)
      .subscribe(
        data => {
          this.connected = data.result;
          this.connecting = false;
          // this.startInferenceLoop()
        },
        error => {
          alert('Connect failed');
          this.connected = false;
          this.connecting = false;
        },
        () => {
          // finish
          this.connecting = false;
        }
      );
  }
  
  startInferenceLoop() {
	  this.inferenceService.inferenceLoop(this.ipAddress, this.interval);
  }
	
}
