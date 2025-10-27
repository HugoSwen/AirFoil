import ast
import os
import re
from typing import List
import openai
import numpy as np
import pandas as pd

from CST import CSTAirfoilParameterization
from prompt import Initialize_prompt_A, Initialize_prompt_B, Initialize_prompt_C, \
    Evolve_prompt_1, Evolve_prompt_2, Evolve_prompt_3, Evolve_prompt_4


class LLMSymbolicRegression:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url='https://api.siliconflow.cn/v1',
        )
        self.model = "Qwen/Qwen3-32B"
        self.population = []
        self.fitness_errors = []

    def llm_query(self, prompt: str, model: str = "Qwen/Qwen3-32B", temperature: float = 0.8):
        """
        向LLM发送查询并返回响应
        """
        if model == "Qwen/Qwen3-32B" or model == "Qwen/Qwen3-14B" or model == "Qwen/Qwen3-8B" or model == "zai-org/GLM-4.6" or model == "Qwen/Qwen3-235B-A22B":
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                extra_body={"enable_thinking": False}
            )
        else:
             response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )

        print(f"LLM Response: {response.choices[0].message.content}")

        return response.choices[0].message.content

    def extract_function_code(self, llm_response: str) -> str:
        """
        从LLM响应中提取Python函数代码
        """
        # 尝试从markdown代码块中提取代码
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            function_code = match.group(1).strip()
        else:
            function_code = llm_response.strip()

        function_code = function_code.replace("return", "").strip()
        try:
            ast.parse(function_code, mode='eval')
            # 校验函数值是否在[0, 1] 范围内，并且是否单调递增
            test_x = np.linspace(0, 1, 100)
            test_y = eval(function_code, {'x': test_x, 'np': np})

            if not (test_y[0] == 0.0 and test_y[-1] == 1.0 and np.all(test_y >= 0) and np.all(test_y <= 1) and np.all(np.diff(test_y) > 0)):
                raise ValueError("Function output not in [0, 1] or not monotonically increasing")
        except Exception as e:
            # raise ValueError(f"Invalid function expression syntax: {llm_response}") from e
            print(
                f"Invalid function expression syntax: {function_code}")
            function_code = "x"

        return function_code

    # 初始化种群
    def initialize(self, population_size: int = 15, prompt_level: str = "B", model: str = "Qwen/Qwen3-32B"):
        """
        根据不同的提示级别初始化种群

        Args:
            population_size: 种群大小
            prompt_level: 提示级别 A/B/C
                A: 基础提示
                B: 添加数学洞察
                C: 添加先前表现良好的函数
        """

        print("Initializing population...")
        print(f"Prompt level: {prompt_level}")

        # 基础初始化提示
        prompt = Initialize_prompt_A

        # 根据实验级别添加数学洞察
        if prompt_level == "B":
            prompt = Initialize_prompt_B

        # 对于实验C，添加先前表现良好的函数
        if prompt_level == "C":
            prompt = Initialize_prompt_C.format(
                functions="\n".join(self.population))

        population = []
        for _ in range(population_size):
            print(f"Generating individual {_ + 1}/{population_size}")
            response = self.llm_query(prompt, model=model)
            function_code = self.extract_function_code(response)
            population.append(function_code)

        self.population = population

        print(f"Initialized population: {self.population}")

        return population

    # 评估函数
    def evaluate(self, functions: List[str],
                 airfoil_data_path: str,
                 order=8,
                 N1=0.5,
                 N2=1.0,
                 is_upper=True):
        """
        评估函数的适应度（拟合误差）

        Args:
            function_code: 要评估的函数代码
            airfoil_data_path: 翼型数据集路径
        """

        print("Evaluating functions...")

        # 提取翼型数据
        fitting_error = []
        airfoil_data = pd.read_csv(airfoil_data_path, header=None)

        for func in functions:
            try:
                # 动态创建并执行函数
                t_func = self.create_executable_function(func)

                # 计算加权误差
                x_original, z_original, z_pred = self.call_cst(
                    t_func, airfoil_data, order, N1, N2, is_upper)

                weights = np.where(x_original < 0.2, 2.0, 1.0)
                fitting_error.append(
                    np.sum(weights * np.abs(z_original - z_pred)))

            except Exception as e:
                # 对于无效函数返回一个很大的数值
                print(f"Error evaluating function {func}: {e}")
                fitting_error.append(float('inf'))

        return fitting_error

    def create_executable_function(self, function_code: str):
        """
        将函数代码字符串转换为可执行函数
        """
        return lambda x: eval(function_code, {'x': x, 'np': np})

    def call_cst(self, t_func, airfoil_data, order=8, N1=0.5, N2=1.0, is_upper=True):
        """
        调用CST预测翼型坐标
        """
        # TODO: 需要根据实际数据格式进行调整(写个函数读取翼型数据)
        af_x = airfoil_data.iloc[:, 0].to_numpy()
        af_z = airfoil_data.iloc[:, 1].to_numpy()

        af_x_upper = af_x[:42]
        af_z_upper = af_z[:42]
        af_x_lower = af_x[42:]
        af_z_lower = af_z[42:]

        cst = CSTAirfoilParameterization(
            order=order, N1=N1, N2=N2, t_func=t_func)

        if is_upper:
            cst.fit_airfoil(af_x_upper, af_z_upper)
            af_z_upper_pred = cst.generate_airfoil(af_x_upper)
            return af_x_upper, af_z_upper, af_z_upper_pred
        else:
            cst.fit_airfoil(af_x_lower, af_z_lower, is_upper=False)
            af_z_lower_pred = cst.generate_airfoil(af_x_lower, is_upper=False)
            return af_x_lower, af_z_lower, af_z_lower_pred

    # 不同的进化策略

    def evolution(self, functions: List[str], strategy: int, model: str = "Qwen/Qwen3-32B"):
        """
        根据不同的策略生成新函数
        Args:
            functions: 父代函数列表
            strategy: 进化策略编号
                1: 完全不同的形式
                2: 受给定函数启发
                3: 修订给定函数
                4: 改变参数设置
        """
        new_functions = []

        if strategy == 1:
            for i in range(len(functions) - 1):
                prompt = Evolve_prompt_1.format(
                    function_1=functions[i],
                    function_2=functions[i+1]
                )
                response = self.llm_query(prompt, model=model)
                new_functions.append(self.extract_function_code(response))

        elif strategy == 2:
            for i in range(len(functions) - 1):
                prompt = Evolve_prompt_2.format(
                    function_1=functions[i],
                    function_2=functions[i+1]
                )
                response = self.llm_query(prompt, model=model)
                new_functions.append(self.extract_function_code(response))

        elif strategy == 3:
            for func in functions:
                prompt = Evolve_prompt_3.format(function=func)
                response = self.llm_query(prompt, model=model)
                new_functions.append(self.extract_function_code(response))

        elif strategy == 4:
            for func in functions:
                prompt = Evolve_prompt_4.format(function=func)
                response = self.llm_query(prompt, model=model)
                new_functions.append(self.extract_function_code(response))

        return new_functions

    def evolve(self, population: List[str], model: str = "Qwen/Qwen3-32B"):
        """
        使用四种进化策略生成新一代种群
        """

        print("Evolving population...")

        new_population = []

        # 策略1: 完全不同的形式
        print("Generating with strategy 1...")
        new_population.extend(self.evolution(population, 1, model=model))

        # 策略2: 受给定函数启发
        print("Generating with strategy 2...")
        new_population.extend(self.evolution(population, 2, model=model))

        # 策略3: 修订给定函数
        print("Generating with strategy 3...")
        new_population.extend(self.evolution(population, 3, model=model))

        # 策略4: 改变参数设置
        print("Generating with strategy 4...")
        new_population.extend(self.evolution(population, 4, model=model))

        return new_population

    # 种群管理
    def manage(self, population: List[str],
               population_size: int,
               airfoil_data_path: str,
               order=8,
               N1=0.5,
               N2=1.0,
               is_upper=True):
        """
        管理种群大小，保留表现最好的个体
        """

        print("Managing population...")

        fitness_errors = self.evaluate(population,
                                       airfoil_data_path,
                                       order,
                                       N1,
                                       N2,
                                       is_upper)

        sorted_indices = np.argsort(fitness_errors)
        managed_population = [population[i]
                              for i in sorted_indices[:population_size]]
        managed_fitness_errors = [fitness_errors[i]
                                  for i in sorted_indices[:population_size]]

        return managed_population, managed_fitness_errors

    def run(self, airfoil_data_path: str,
            num_generations: int = 20,
            population_size: int = 15,
            prompt_level: str = "B",
            model: str = "Qwen/Qwen3-32B",
            order=8,
            N1=0.5,
            N2=1.0,
            is_upper=True):
        """
        运行完整的LLM符号回归进化流程
        """
        print("Starting LLM-based symbolic regression...")

        # 1. 初始化种群
        self.initialize(population_size, prompt_level, model=model)

        for generation in range(num_generations):
            print(f"Generation {generation + 1}/{num_generations}")

            # 2. 评估
            self.fitness_errors = self.evaluate(
                self.population, airfoil_data_path, order, N1, N2, is_upper)

            # 3. 进化
            current_population = self.population
            current_population.extend(
                self.evolve(self.population, model=model))

            # 4. 种群管理
            self.population, self.fitness_errors = self.manage(current_population,
                                                               population_size,
                                                               airfoil_data_path,
                                                               order,
                                                               N1,
                                                               N2,
                                                               is_upper)

            # 5. 更新种群 (精英保留策略)
            print(f"Current best fitness: {self.fitness_errors[0]:.6f}")

        return self.population[0], self.fitness_errors[0]
