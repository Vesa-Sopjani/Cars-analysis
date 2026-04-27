import { Component } from '@angular/core';
import { Header } from '../common/header/header';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-index',
  imports: [Header, RouterLink],
  templateUrl: './index.html',
  styleUrl: './index.scss',
})
export class Index {}
