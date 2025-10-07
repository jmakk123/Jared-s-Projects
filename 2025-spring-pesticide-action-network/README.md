# 2025-spring-pesticide-action-network

## Project Background
The California People and Pesticide Explorer is a tool developed by the DSI Open Spatial Lab and 11th Hour teams working with Pesticide Action Network (PAN) and Californians for Pesticide Reform (CPR). The tool makes the highly granular Pesticide Use Regulation (PUR) data more easily accessible and viewable in a dynamic interactive web application. This tool builds on existing workflows by enabling live filter, aggregation, and data downloads, along with incorporating key demographic data to help understand the social context of who gets exposed to pesticide use. CPR and PAN use this tool and others to help identify problematic usage of harmful chemicals - those with carcinogenic properties or posing reproductive risks. Demographic data and census spatial units (Tracts, School Districts, Zip Code Tabulation Areas) are important to help identify which communities might bear a disproportionate exposure to certain active ingredients, and which at risk communities (children, elderly) may need advocacy and support to reform pesticide use in their areas.

Methodology can be found on the About tab of the [live interactive web app](https://pesticideinfo.org/pesticide-maps/ca-pesticide-map).

## Team Members
- Jenny Li (jennyyueli@uchicago.edu)
- Emma Griffin (griffin13@uchicago.edu)
- Liam Starnes (lstarnes@uchicago.edu)
- Jared Maksoud (jmaksoud@uchicago.edu)

## Project Goals
The existing API allows for easy querying of the data. This analysis will utilize the data to study specific active ingredient use in key communities:

1. Which communities have the most exposure to different active ingredients? Do any socio-economic factors suggest a disproportionate impact to BIPOC communities?
2. For certain harmful active ingredients (specifics TBD), which schools and school districts see the most use?
3. Over the past five years, have any potential trends and usage near schools emerged? 

To get started answering these questions, read `notebokos/pesticide_docs.py` which contains metadata and sample usage. Start with the following tasks:
- Find the county which has the highest pesticide usage and make a map of the school districts within it
- Find school locations, then identify the "Sections" within a 2 mile radius. Query the data and summarize the sum of usage nearby schools.
- Research which pesticide active ingredients and health concern are most important. Create a list of the top 10 school districts exposed to those chemicals and health concerns. 
## Usage

See `notebokos/pesticide_docs.py` to get started.
### Docker

### Docker & Make

We use `docker` and `make` to run our code. There are three built-in `make` commands:

* `make build-only`: This will build the image only. It is useful for testing and making changes to the Dockerfile.
* `make run-notebooks`: This will run a jupyter server which also mounts the current directory into `\program`.
* `make run-interactive`: This will create a container (with the current directory mounted as `\program`) and loads an interactive session. 

The file `Makefile` contains information about about the specific commands that are run using when calling each `make` statement.

### Developing inside a container with VS Code

If you prefer to develop inside a container with VS Code then do the following steps. Note that this works with both regular scripts as well as jupyter notebooks.

1. Open the repository in VS Code
2. At the bottom right a window may appear that says `Folder contains a Dev Container configuration file...`. If it does, select, `Reopen in Container` and you are done. Otherwise proceed to next step. 
3. Click the blue or green rectangle in the bottom left of VS code (should say something like `><` or `>< WSL`). Options should appear in the top center of your screen. Select `Reopen in Container`.




## Style
We use [`ruff`](https://docs.astral.sh/ruff/) to enforce style standards and grade code quality. This is an automated code checker that looks for specific issues in the code that need to be fixed to make it readable and consistent with common standards. `ruff` is run before each commit via [`pre-commit`](https://pre-commit.com/). If it fails, the commit will be blocked and the user will be shown what needs to be changed.

To check for errors locally, first ensure that `pre-commit` is installed by running `pip install pre-commit` followed by `pre-commit install`. Once installed, check for errors by running:
```
pre-commit run --all-files
```

## Repository Structure

### utils
Project python code

### notebooks
Contains short, clean notebooks to demonstrate analysis.

### data

Contains details of acquiring all raw data used in repository. If data is small (<50MB) then it is okay to save it to the repo, making sure to clearly document how to the data is obtained.

If the data is larger than 50MB than you should not add it to the repo and instead document how to get the data in the README.md file in the data directory. 

This [README.md file](/data/README.md) should be kept up to date.

### output
Should contain work product generated by the analysis. Keep in mind that results should (generally) be excluded from the git repository.
