import { Component } from '@angular/core';
import { Header } from '../common/header/header';

@Component({
  selector: 'app-index',
  imports: [Header],
  templateUrl: './index.html',
  styleUrl: './index.scss',
})
export class Index {}
