# thesis_template
This is a template for Master and Bachelor theses code at LSX

## recommended stuff
- [Hydra for managing configs](https://hydra.cc/)
- [Pytorch Lightning for boilerplate training code](https://lightning.ai/)
- [WandB for logging](https://wandb.ai/)

## Setting up environment

It is recommended to use a virtual environment for Python projects to manage dependencies. Here are the steps to create a new virtual environment:

1. Install the virtual environment package if not already installed. Open your terminal and run:
    ```bash
    pip install virtualenv
    ```

2. Navigate to your project directory:
    ```bash
    cd /path/to/your/project
    ```

3. Create a new virtual environment in your project directory:
    ```bash
    virtualenv venv
    ```

4. Activate the virtual environment:
    - On Linux or MacOS, run:
        ```bash
        source venv/bin/activate
        ```
    - On Windows, run:
        ```cmd
        .\venv\Scripts\activate
        ```

5. Now your virtual environment is activated and you can install your project's dependencies in it. It is recommended to store all your dependencies in a single requirements.txt file:
    ```bash
    pip install -r requirements.txt
    ```

Remember to activate the virtual environment every time you work on your project. When you're done, you can deactivate it by simply typing `deactivate` in the terminal.
If you use an IDE like VScode (recommended) or PyCharm, it can manage your environment automatically.