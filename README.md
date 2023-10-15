# PhD
All works related to my PhD. Coming soon...

<details>
<summary>Setting Up a Virtual Environment for Linux</summary>

Follow these steps to set up a virtual environment for running the project:

1. **Create a working directory and navigate into it:**

    ```bash
    mkdir project-env && cd project-env
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv virtual-environment
    ```

3. **Activate the virtual environment:**

    ```bash
    source virtual-environment/bin/activate
    ```

4. **Verify that there are no packages installed running `pip list --local`:**

    ```bash
    (venv) user@your-machine:~/project-env$ pip list --local
    Package    Version
    ---------- -------
    pip        22.0.2
    setuptools 59.6.0
    ```

5. **Install the necessary packages from the `requirements.txt` file:**

    ```bash
    pip install -r requirements.txt
    ```

6. **Verify that the required packages are installed:**

    ```bash
    pip list --local
    ```

Now your virtual environment is set up and ready to run the project. Enjoy coding!

</details>
