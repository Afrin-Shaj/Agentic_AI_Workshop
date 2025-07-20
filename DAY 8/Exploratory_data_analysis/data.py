#!/usr/bin/env python3
"""
Sample Dataset Generator for Multi-Agent EDA System
==================================================

This script creates sample datasets for testing the EDA system.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_datasets():
    """Create various sample datasets for testing."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Dataset 1: Customer Analysis Data
    print("Creating customer_data.csv...")
    n_customers = 1000
    
    customer_data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(35, 12, n_customers).astype(int),
        'income': np.random.exponential(50000, n_customers),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_customers, p=[0.3, 0.4, 0.2, 0.1]),
        'satisfaction_score': np.random.randint(1, 11, n_customers),
        'months_subscribed': np.random.exponential(12, n_customers),
        'total_spent': np.random.gamma(2, 500, n_customers),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04]),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
        'churn': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
    }
    
    # Add some correlations
    customer_data['income'] = np.where(
        customer_data['education'] == 'PhD', 
        customer_data['income'] * 1.5,
        customer_data['income']
    )
    
    # Add missing values
    missing_indices = np.random.choice(n_customers, int(0.05 * n_customers), replace=False)
    customer_data['income'][missing_indices] = np.nan
    
    df_customers = pd.DataFrame(customer_data)
    df_customers.to_csv('data/customer_data.csv', index=False)
    
    # Dataset 2: Sales Data
    print("Creating sales_data.csv...")
    n_sales = 2000
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(365)]
    
    sales_data = {
        'date': np.random.choice(dates, n_sales),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_sales),
        'sales_amount': np.random.lognormal(4, 1, n_sales),
        'quantity_sold': np.random.poisson(3, n_sales) + 1,
        'discount_percent': np.random.exponential(10, n_sales),
        'store_location': np.random.choice(['North', 'South', 'East', 'West'], n_sales),
        'salesperson_id': np.random.randint(1, 51, n_sales),
        'customer_rating': np.random.normal(4.2, 0.8, n_sales)
    }
    
    # Ensure ratings are between 1 and 5
    sales_data['customer_rating'] = np.clip(sales_data['customer_rating'], 1, 5)
    
    # Add seasonal effects
    sales_data['month'] = [d.month for d in sales_data['date']]
    holiday_boost = np.where(np.isin(sales_data['month'], [11, 12]), 1.3, 1.0)
    sales_data['sales_amount'] = sales_data['sales_amount'] * holiday_boost
    
    df_sales = pd.DataFrame(sales_data)
    df_sales.to_csv('data/sales_data.csv', index=False)
    
    # Dataset 3: Iris-like Dataset (Classic ML dataset)
    print("Creating iris_sample.csv...")
    n_flowers = 150
    
    # Create three species with different characteristics
    species_data = []
    for species, params in [
        ('Setosa', {'sepal_length': (5.0, 0.3), 'sepal_width': (3.4, 0.3), 'petal_length': (1.4, 0.2), 'petal_width': (0.2, 0.1)}),
        ('Versicolor', {'sepal_length': (5.9, 0.5), 'sepal_width': (2.8, 0.3), 'petal_length': (4.3, 0.5), 'petal_width': (1.3, 0.2)}),
        ('Virginica', {'sepal_length': (6.6, 0.6), 'sepal_width': (3.0, 0.3), 'petal_length': (5.6, 0.6), 'petal_width': (2.0, 0.3)})
    ]:
        for _ in range(50):
            row = {
                'sepal_length': np.random.normal(params['sepal_length'][0], params['sepal_length'][1]),
                'sepal_width': np.random.normal(params['sepal_width'][0], params['sepal_width'][1]),
                'petal_length': np.random.normal(params['petal_length'][0], params['petal_length'][1]),
                'petal_width': np.random.normal(params['petal_width'][0], params['petal_width'][1]),
                'species': species
            }
            species_data.append(row)
    
    df_iris = pd.DataFrame(species_data)
    df_iris.to_csv('data/iris_sample.csv', index=False)
    
    # Dataset 4: Employee Data
    print("Creating employee_data.csv...")
    n_employees = 500
    
    employee_data = {
        'employee_id': [f'EMP{str(i).zfill(4)}' for i in range(1, n_employees + 1)],
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Operations'], n_employees),
        'years_experience': np.random.exponential(5, n_employees),
        'salary': np.random.normal(75000, 20000, n_employees),
        'performance_rating': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], n_employees, p=[0.15, 0.35, 0.4, 0.1]),
        'remote_work': np.random.choice(['Yes', 'No'], n_employees, p=[0.4, 0.6]),
        'training_hours': np.random.poisson(20, n_employees),
        'job_satisfaction': np.random.randint(1, 11, n_employees),
        'overtime_hours': np.random.exponential(10, n_employees)
    }
    
    # Add correlations
    experience_salary_correlation = 2000 * employee_data['years_experience']
    employee_data['salary'] = employee_data['salary'] + experience_salary_correlation
    
    df_employees = pd.DataFrame(employee_data)
    df_employees.to_csv('data/employee_data.csv', index=False)
    
    print("\n✅ Sample datasets created successfully!")
    print("\nAvailable datasets:")
    print("1. data/customer_data.csv - Customer analysis with demographics and behavior")
    print("2. data/sales_data.csv - Sales transactions with temporal patterns")
    print("3. data/iris_sample.csv - Flower measurements (classic ML dataset)")
    print("4. data/employee_data.csv - Employee performance and satisfaction")
    
    print("\nUsage examples:")
    print("python main.py --data data/customer_data.csv")
    print("python main.py --data data/sales_data.csv --verbose")
    print("python main.py --data data/iris_sample.csv --output results")
    
    return [
        'data/customer_data.csv',
        'data/sales_data.csv', 
        'data/iris_sample.csv',
        'data/employee_data.csv'
    ]

if __name__ == "__main__":
    datasets = create_sample_datasets()
    
    # Show basic info about created datasets
    print("\n📊 Dataset Information:")
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"\n{dataset_path}:")
            print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"  - Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            print(f"  - Missing values: {df.isnull().sum().sum()}")