# Forecasting
Machine learning models that produce rolling forecasts

## Setting Up
1. Install Ubuntu 18.04 from the Microsoft Store and run it

2. Install VSCode for Windows

3. Open up GitHub and create a repository initialized with a README.md file

4. Install pip and virtualenv
```ssh
sudo apt update
sudo apt install python3-pip
sudo pip3 install virtualenv
```

5. Configure GitHub credentials
```ssh
git config --global user.name "Name"
git config --global user.email "Email"
```

6. Clone this repository
```ssh
mkdir repos
cd repos
git clone https://github.com/nmorris-fcx/Forecasting.git
cd Forecasting
```

7. Create a virtual enviornment and code
```ssh
python3 -m virtualenv mlenv
source mlenv/bin/activate
code .
```

## Initializing the Repository
1. Save GitHub username and password
```ssh
git config credential.helper store
git pull
```

2. Go to the main branch and create a new branch off of it (your new working directory)
```ssh
git checkout main
git checkout -b initialize
git push origin initialize
```

3. Create an issue in GitHub to update the README file

4. In VSCode update the README.md file with procedures and push this change to the new branch
```ssh
git add README.md
git commit -m "Adding information to README that explains how to set up your local machine to communicate with GitHub, #1"
git push origin initialize
```

5. Create an issue in GitHub to add a requirements file

6. In VSCode create a requirements.txt file
```ssh
git add .
git commit -m "Adding a requirements file to install python modules, #2"
git push origin initialize
```

7. Create an issue in GitHub to add source code and test code

8. In VSCode create a src folder and a test folder for building the forecasting module, and add an empty __init__.py file to each
```ssh
git add .
git commit -m "Adding first version of source and test code, #3"
git push origin initialize
```

9. In GitHub create a pull request to merge initialize into main, then do a code review before merging

10. After merging the pull request using GitHub, close the issues

11. (skip this step if you merged the pull request using GitHub) How to merge initialize into main using ubuntu
```ssh
git checkout main
git merge initialize
git push
```

12. Update your local version of main
```ssh
git checkout main
git pull
```

12. How to delete a branch from GitHub, and then from your local
```ssh
git checkout main
git push origin --delete initialize
git branch -d initialize
```

## Building the Repository
1. Go to the main branch and create a new branch off of it (your new working directory)
```ssh
git checkout main
git checkout -b forecast_class
git push origin forecast_class
```

2. How to install requirements.txt
```ssh
pip install -r requirements.txt
```

3. Create an issue in GitHub to add a forecasting class

4. Write and commit code
```ssh
git add .
git commit -m "Completing the first version of the Forecasting class for modeling, #5"
git push origin forecast_class
```

5. In GitHub create a pull request to merge forecast_class into main, then do a code review before merging and closing issues