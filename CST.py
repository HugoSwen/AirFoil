import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import comb

class CSTAirfoilParameterization:
    """
    CST 翼型参数化方法
    """
    
    def __init__(self, order=8, N1=0.5, N2=1.0, t_func=lambda x: x):
        """
        order: Bernstein多项式的阶数
        N1, N2: 类别函数的指数参数
        t_func: 特征变换函数 t(x)，默认为恒等函数 t(x)=x
        """
        self.order = order
        self.N1 = N1
        self.N2 = N2
        
        # 设置特征变换函数（默认为恒等变换）
        self.t_func = t_func

        self.upper_coeffs = None
        self.lower_coeffs = None
        self.z_te_upper = 0.0
        self.z_te_lower = 0.0
    
    def class_function(self, x):
        """
        类别函数 C(t) = t^N1 * (1-t)^N2, 
        其中 t = t(x)
        """
        t = self.t_func(x)
        return t**self.N1 * (1-t)**self.N2
    
    def bernstein_polynomial(self, i, n, x):
        """
        计算第i个n阶Bernstein多项式基函数
        B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)
        其中 t = t(x)
        """
        t = self.t_func(x)
        return comb(n, i) * (t**i) * ((1-t)**(n-i))
    
    def shape_function(self, x, coefficients):
        """
        形状函数 S(t) = Σ A_i * B_i^n(t), 
        其中 t = t(x)
        """
        n = len(coefficients) - 1
        result = np.zeros_like(x)
        for i, coeff in enumerate(coefficients):
            result += coeff * self.bernstein_polynomial(i, n, x)
        return result
    
    def airfoil_surface(self, x, coefficients, z_te):
        """
        计算翼型表面坐标
        z(x) = C(t) * S(t) + x * z_te, 
        其中 t = t(x)
        """
        C_t = self.class_function(x)
        S_t = self.shape_function(x, coefficients)
        
        return C_t * S_t + x * z_te

    # TODO: 需要确认z_te的取值
    def fit_airfoil(self, x, z, is_upper=True):
        """
        使用最小二乘法拟合翼型数据
        """
        # 确保x在[0,1]范围内(数据已归一化)

        # 初始猜测系数
        n_coeffs = self.order + 1
        initial_guess = np.ones(n_coeffs) * 0.1
        z_te = z[-1]
        
        # 定义error函数
        def error(coefficients):
            z_pred = self.airfoil_surface(x, coefficients, z_te)
            return z_pred - z
        
        # 最小二乘拟合权重
        result = least_squares(error, initial_guess, method='trf')
        
        if is_upper:
            self.upper_coeffs = result.x
            self.z_te_upper = z_te
        else:
            self.lower_coeffs = result.x
            self.z_te_lower = z_te

    def generate_airfoil(self, x, is_upper=True):
        """
        生成完整的翼型坐标
        """
        if is_upper:
            if self.upper_coeffs is None:
                raise ValueError("Upper surface coefficients not fitted yet.")
            z_upper_pred = self.airfoil_surface(x, self.upper_coeffs, self.z_te_upper)
            return z_upper_pred
        else:
            if self.lower_coeffs is None:
                raise ValueError("Lower surface coefficients not fitted yet.")
            z_lower_pred = self.airfoil_surface(x, self.lower_coeffs, self.z_te_lower)
            return z_lower_pred
    
    def plot_airfoil(self, x_upper, z_upper, x_lower, z_lower, z_upper_pred, z_lower_pred):
        """
        绘制翼型及其拟合结果
        """
        plt.style.use('default')
        plt.figure(figsize=(6, 3))
        plt.plot(x_upper, z_upper, 'g-', label='Upper Surface Data')
        plt.plot(x_lower, z_lower, 'g-')
        plt.plot(x_upper, z_upper_pred, 'b-', label='Fitted Upper Surface')
        plt.plot(x_lower, z_lower_pred, 'b-')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('CST Airfoil Parameterization')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    def plot_residuals(self, x_upper, z_upper, x_lower, z_lower, z_upper_pred, z_lower_pred):
        """
        绘制翼型拟合的残差图
        """
        # 1. 计算残差 (真实值 - 预测值)
        upper_residuals = z_upper - z_upper_pred
        lower_residuals = z_lower - z_lower_pred

        # 2. 绘图
        plt.style.use('default') # 确保是白色背景
        fig, ax = plt.subplots(figsize=(6, 3))

        # 绘制上下翼面的残差
        # 使用 'o-' 样式来同时显示数据点和连接线
        ax.plot(x_upper, upper_residuals, '.-', color='black', label='Upper Surface Residuals', markersize=4)
        ax.plot(x_lower, lower_residuals, '.-', color='red', label='Lower Surface Residuals', markersize=4)

        # 添加一条 y=0 的水平参考线
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        # 设置标签、标题和图例
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('z residuals', fontsize=10)
        ax.legend()
        ax.grid(True)
        
        # 固定z坐标范围
        ax.set_ylim(-0.0015, 0.0015)
        
        plt.show()
