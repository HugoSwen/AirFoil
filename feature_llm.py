import os
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
        self.population = []
        self.fitness_values = []

    def llm_query(self, prompt: str, model: str = "Qwen/Qwen3-32B", temperature: float = 0.8):
        """
        向LLM发送查询并返回响应
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content

    def extract_function_code(self, llm_response: str) -> str:
        """
        从LLM响应中提取Python函数代码
        """
        # TODO: 这里需要实现具体的提取逻辑
        
        pass

    # 初始化种群
    def initialize(self, population_size: int = 15, prompt_level: str = "B"):
        """
        根据不同的提示级别初始化种群
        
        Args:
            population_size: 种群大小
            prompt_level: 提示级别 A/B/C
                A: 基础提示
                B: 添加数学洞察
                C: 添加先前表现良好的函数
        """
    
        # 基础初始化提示
        if prompt_level == "A":
            initialize_prompt = Initialize_prompt_A
        
        # 根据实验级别添加数学洞察
        elif prompt_level == "B":
            initialize_prompt += Initialize_prompt_B

        # 对于实验C，添加先前表现良好的函数
        elif prompt_level == "C":
            well_performed_functions = self.load_well_performed_functions()
            initialize_prompt += Initialize_prompt_C.format(functions="\n".join(well_performed_functions))

        population = []
        for i in range(population_size):
            response = self.llm_query(prompt=initialize_prompt)
            function_code = self.extract_function_code(response.choices[0].message.content)
            population.append(function_code)
        
        self.population = population
        return population

    # 评估函数
    def evaluate(self, function_code: str, airfoil_data_path: str, order=8, N1=0.5, N2=1.0, is_upper=True):
        """
        评估函数的适应度（拟合误差）
        
        Args:
            function_code: 要评估的函数代码
            airfoil_data_path: 翼型数据集路径
        """
        try:
            # 提取翼型数据
            airfoil_data = pd.read_csv(airfoil_data_path, header=None)

            # 动态创建并执行函数
            t_func = self.create_executable_function(function_code)

            # 计算加权误差
            x_original, z_original, z_pred = self.call_cst(t_func, airfoil_data, order, N1, N2, is_upper)

            weights = np.where(x_original < 0.2, 2.0, 1.0)
            fitting_error = np.sum(weights * (z_original - z_pred))

            return fitting_error
            
        except Exception as e:
            # 对于无效函数返回一个很大的数值
            print(f"Error evaluating function {function_code}: {e}")
            return float('inf')

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

        af_x_upper = af_x[:26][::-1]
        af_z_upper = af_z[:26][::-1]
        af_x_lower = af_x[26:]
        af_z_lower = af_z[26:]

        cst = CSTAirfoilParameterization(order=order, N1=N1, N2=N2, t_func=t_func)

        if is_upper:
            cst.fit_airfoil(af_x_upper, af_z_upper)
            af_z_upper_pred = cst.generate_airfoil(af_x_upper)
            return af_x_upper, af_z_upper, af_z_upper_pred
        else:
            cst.fit_airfoil(af_x_lower, af_z_lower, is_upper=False)
            af_z_lower_pred = cst.generate_airfoil(af_x_lower, is_upper=False)
            return af_x_lower, af_z_lower, af_z_lower_pred


    # 不同的进化策略
    def evolution(self, parent_functions: List[str], strategy: int):
        """
        根据不同的策略生成新函数
        Args:
            parent_functions: 父代函数列表
            strategy: 进化策略编号
                1: 完全不同的形式
                2: 受给定函数启发
                3: 修订给定函数
                4: 改变参数设置
        """
        new_functions = []
        
        if strategy == 1:
            for i in range(0, len(parent_functions)-1, 2):
                prompt = Evolve_prompt_1.format(
                    function_1=parent_functions[i],
                    function_2=parent_functions[i+1]
                )
                response = self.llm_query(prompt)
                new_functions.append(self.extract_function_code(response))

        elif strategy == 2:
            for i in range(0, len(parent_functions)-1, 2):
                prompt = Evolve_prompt_2.format(
                    function_1=parent_functions[i],
                    function_2=parent_functions[i+1]
                )
                response = self.llm_query(prompt)
                new_functions.append(self.extract_function_code(response))

        elif strategy == 3:
            for func in parent_functions[:3]:  # 取前3个最好的函数
                prompt = Evolve_prompt_3.format(function=func)
                response = self.llm_query(prompt)
                new_functions.append(self.extract_function_code(response))

        elif strategy == 4:
            for func in parent_functions[:3]:
                prompt = Evolve_prompt_4.format(function=func)
                response = self.llm_query(prompt)
                new_functions.append(self.extract_function_code(response))

        return new_functions
    
    def evolve(self, current_population: List[str], fitness_scores: List[float]):
        """
        使用四种进化策略生成新一代种群
        """
        new_population = []
        
        # 选择表现最好的个体作为父代
        sorted_indices = np.argsort(fitness_scores)
        best_functions = [current_population[i] for i in sorted_indices[:5]]
        
        # 策略1: 完全不同的形式
        new_population.extend(self.evolution(best_functions, 1))
        
        # 策略2: 受给定函数启发
        new_population.extend(self.evolution(best_functions, 2))

        # 策略3: 修订给定函数
        new_population.extend(self.evolution(best_functions, 3))

        # 策略4: 改变参数设置
        new_population.extend(self.evolution(best_functions, 4))

        # 排序并截取前N个作为新一代种群
        new_population = sorted(new_population, key=lambda f: self.evaluate(f, airfoil_data_path="path/to/airfoil.csv"))

        return new_population[:len(current_population)]  # 保持种群大小不变
    

    def run_evolution(self, airfoil_data_path: str, 
                      num_generations: int = 20, 
                      population_size: int = 15, 
                      prompt_level: str = "B", 
                      order=8, 
                      N1=0.5, 
                      N2=1.0):
        """
        运行完整的LLM符号回归进化流程
        """
        print("Starting LLM-based symbolic regression...")
        
        # 1. 初始化种群
        self.initialize(population_size, prompt_level)
        
        best_function = None
        best_fitting_error = float('inf')
        
        for generation in range(num_generations):
            print(f"Generation {generation + 1}/{num_generations}")
            
            # 2. 评估
            
            # 3. 进化

            # 4. 种群管理
            
            # 5. 更新种群 (精英保留策略)
            print(f"Best fitness: {best_fitting_error:.2f}")

        return best_function, best_fitting_error
