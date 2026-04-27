import { Routes } from '@angular/router';
import { Index } from './components/index/index';
import { AnalyticsDashboard } from './components/analytics-dashboard/analytics-dashboard';
import { PriceEvaluator } from './components/price-evaluator/price-evaluator';
import { PriceSuggester } from './components/price-suggester/price-suggester';
import { CarComparison } from './components/car-comparison/car-comparison';

export const routes: Routes = [
  {
    path: 'analytics-dashboard',
    component: AnalyticsDashboard,
  },
  {
    path: 'price-evaluator',
    component: PriceEvaluator,
  },
  {
    path: 'price-suggester',
    component: PriceSuggester,
  },
  {
    path: 'car-comparison',
    component: CarComparison,
  },
  {
    path: '',
    component: Index,
  },
];
