"""
字体工具模块
解决matplotlib中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib
import platform
import os

def setup_chinese_font():
    """设置中文字体支持"""
    system = platform.system()
    
    if system == "Windows":
        # Windows系统字体
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'DejaVu Sans']
    elif system == "Darwin":  # macOS
        # macOS系统字体
        fonts = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'DejaVu Sans']
    else:  # Linux
        # Linux系统字体
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
    
    # 尝试设置字体大小
    try:
        plt.rcParams['font.size'] = 10
    except:
        pass
    
    # 清除matplotlib字体缓存
    try:
        from matplotlib.font_manager import _rebuild
        _rebuild()
    except:
        pass

def setup_english_font():
    """设置英文字体（避免中文字体问题）"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

def get_available_fonts():
    """获取系统可用字体"""
    try:
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        fonts = [f.name for f in fm.ttflist]
        return sorted(set(fonts))
    except:
        return []

def check_chinese_font_support():
    """检查中文字体支持"""
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试', fontsize=12)
        plt.close(fig)
        return True
    except:
        return False

def setup_robust_font():
    """设置健壮的字体配置，支持中英文混合"""
    system = platform.system()
    
    # 设置字体优先级列表
    if system == "Windows":
        fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans', 'Arial']
    elif system == "Darwin":  # macOS
        fonts = ['PingFang SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'DejaVu Sans', 'Arial']
    else:  # Linux
        fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans', 'Arial']
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    # 设置matplotlib后端
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
    except:
        pass

def safe_plot_setup(use_chinese=True):
    """安全的绘图设置"""
    try:
        if use_chinese:
            setup_robust_font()
            # 测试中文字体是否可用
            if not check_chinese_font_support():
                print("Warning: Chinese font not available, falling back to English")
                setup_english_font()
        else:
            setup_english_font()
    except Exception as e:
        print(f"Font setup failed: {e}, using default settings")
        setup_english_font()
    
    # 设置matplotlib后端以避免GUI问题
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
    except:
        pass

# 自动设置字体
if __name__ == "__main__":
    print("Available fonts:")
    fonts = get_available_fonts()
    for font in fonts[:10]:  # 显示前10个字体
        print(f"  - {font}")
    
    print(f"\nChinese font support: {check_chinese_font_support()}")
    
    # 测试字体设置
    safe_plot_setup(use_chinese=True)
