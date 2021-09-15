## Project Goals:
>    - Create documented files to clean and prepare Zillow dataset for processing by regression ML algorithms.
>    - Use ML algorithms to create a model that best predicts tax value of properties on both in and out-of-sample data.
>    - Evaluate each iteration of my models while changing parameters and features to find the best model in terms of RMSE and R^2 score.
>    - Document processes, findings, and takeaways in a draft Jupyter Notebook
>    - Present on my final Jupyter Notebook in a canva slide deck, giving a high-level overview of the process used to create the model of best fit as well as basic evaluation metrics of the model.



## Data dictionary
Target  | Description   | Data Type
--|--|--
tax_value    | The tax value of a property | float64

Categorical Features   | Description |    Data Type
--|--|--
bedrooms    |   Count of bedrooms per property | float64
bathrooms    |   Count of bathrooms per property | float64
year_built |    Year a home was constructed    | object
taxamount |    Amount paid in taxes so far   | float64
fips |        Numeric county code    | object


Continuous Features | Description | Data Type
--|--|--
area | Internal square footage of a home | float64

Engineered Features  | Description   | Data Type
--|--|--
county |    Derrived from fips, denotes the actual county of a home    | object
month_sold |    Derrived from sale_date (dropped) and indicates the month a home was sold    | int64
tax_rate |    tax rate of a property, derrived from (taxamount / tax_value) * 100 |    float64



## Hypotheses:
>   - $H_{i}$: Bedroom count and bathroom count are the main drivers of tax_value


## Plan:
- [x] Create repo on github to save all files related to the project.
- [x] Create README.md with [x] goals, [x] initial hypotheses, [x] data dictionary, and [x] outline plans for the project.
- [x] Acqiure zillow data using acquire.py file drawing directly from Codeups `zillow` database with SQL queries. Create functions for use in conjunction with prepare.py.
- [x] Clean, tidy, and encode data in such a way that it is usable in a machine learning algorithm. Includes dropping unneccesary columns, creating dummies where needed and changing string values to numeric values and getting rid of outliers
- [x] Create hypotheses based on preliminary statistical tests
- [x] Test hypotheses with tests such as t-test, chi-squared to determine the viability of said hypotheses by comparing p-values to alpha.
- [x] Establish a baseline accuracy.
- [x] Train three different classification models from OLS, GLM, and Lasso + Lars, testing a variety of parameters and features, both engineered and pre-existing.
- [x] Evaluate models using RMSE, R^2 score, and other metrics on in-sample and out-of-sample datasets.
- [x] Once a single, best preforming model has been chosen, evaluate the preformance of the model on the test dataset.
- [x] Create a slide deck with all of my deliverables.
- [] Present my slide deck to Codeup instructors