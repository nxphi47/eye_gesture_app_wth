import {Injectable} from '@angular/core';
import {HttpClient, HttpHeaders} from "@angular/common/http";
import { Http } from "@angular/http";
import {Observable} from "rxjs/Observable";
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/toPromise';
import 'rxjs/add/operator/timeout';

export interface InferenceResponse {
  prob: number,
  label: string,
  label_set: string[]
}

export interface PingResponse {
  result: boolean
}

@Injectable()
export class InferenceService {
  ipAddress: string = '';
  instantResponse: InferenceResponse = null;
  looping:  boolean = false;
  constructor(private httpClient: HttpClient, private http: Http) {
  
  }
  
  public inference(ipAddress: string) {
    this.ipAddress = ipAddress;
    return this.httpClient.get<InferenceResponse>(`http://${ipAddress}/inference`)
      .timeout(2000)
      ;
  }
  
  public get response() {
    return this.instantResponse;
  }
  
  // interval in mili second
  public inferenceLoop(ipAddress: string, interval: number = 10, callback?: any) {
    this.ipAddress = ipAddress;
    this.inference(ipAddress)
      .subscribe(
        res => {
          this.instantResponse = res;
          console.log(res);
          if (callback) {
            callback(this.instantResponse);
          }
          if (this.instantResponse && this.instantResponse.label) {
            setTimeout(() => {
              this.inferenceLoop(ipAddress, interval);
            }, interval)
          }
        }
      );
  }
  
  public connect(ipAddress: string) {
    this.ipAddress = ipAddress;
    console.log('Get');
    return this.httpClient.get<PingResponse>(`http://${ipAddress}/ping`)
      .timeout(1000)
  }
  
  public faceInference(ipAddress: string) {
    return new Observable(observer => {
      setTimeout(() => {
        observer.next({
          prob: [0.1, 0.1, 0.2, 0.2, 0.4, 0],
          label: 'double_blink',
          label_set: ['left', 'right', 'up', 'down', 'center', 'double_blink']
        });
        observer.complete();
      })
    });
  }
}
