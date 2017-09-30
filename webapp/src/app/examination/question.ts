export class Question {
  content: string;
  choices: string[];
  answer: number; // 0 to 3
  
  selectedChoice: number;
  constructor(content: string, choices: string[], answer: number) {
    this.content = content;
    this.choices = choices;
    this.answer = answer;
    this.selectedChoice = null;
  }
  
  public validate(ans: number = null): boolean {
    return (ans != null) ? this.answer === ans : this.answer == this.selectedChoice;
  }
  
  public select(c: number) {
    this.selectedChoice = c;
  }
}
