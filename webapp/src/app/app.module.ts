import {BrowserModule} from "@angular/platform-browser";
import {NgModule} from "@angular/core";

import {AppComponent} from "./app.component";
import {InferenceService} from "./services/inference.service";
import {CommonModule} from "@angular/common";
import {BrowserAnimationsModule} from "@angular/platform-browser/animations";
import {FormsModule, ReactiveFormsModule} from "@angular/forms";
import {HttpModule} from "@angular/http";
import {HttpClientModule} from "@angular/common/http";
import {RouterModule} from "@angular/router";
import {HomeComponent} from "./home/home.component";
import {EyeIdComponent} from "./eye-id/eye-id.component";
import {ExaminationComponent} from "./examination/examination.component";
import {TetrixComponent} from "./tetrix/tetrix.component";
// import {MdButtonModule, MdCheckboxModule} from '@angular/material';


@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    EyeIdComponent,
    ExaminationComponent,
    TetrixComponent,
  
  ],
  imports: [
    BrowserModule,
    CommonModule,
    // MaterialImportModule,
    BrowserAnimationsModule,
    FormsModule,
    ReactiveFormsModule,
    HttpModule,
    HttpClientModule,
    // RouterModule.forRoot([
    //   {path: '', redirectTo: '/home', pathMatch: 'full'},
    //   {path: 'home', component: HomeComponent},
    //   {path: 'eyeid', component: EyeIdComponent},
    //   {path: 'exam', component: ExaminationComponent},
    //   {path: 'tetrix', component: TetrixComponent},
    // ])
  ],
  providers: [
    InferenceService
  ],
  bootstrap: [AppComponent]
})
export class AppModule {

}
