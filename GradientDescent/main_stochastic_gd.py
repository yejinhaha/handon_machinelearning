import pandas as pd
from linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def main():
    # CSV 파일 경로 설정
    file_path = './data/Salary_dataset.csv'
    # CSV 파일을 DataFrame으로 로드
    df = pd.read_csv(file_path)
    # DataFrame의 첫 5행 출력
    print(df.head())

    years_experience = df['YearsExperience'].values
    salary = df['Salary'].values

    model = LinearRegression()
    model.set(w=0.3, b=0.2)
    num_data = len(years_experience)
    max_epochs = 50000
    upd_dist = 0.0001

    for e in range(0, max_epochs):
        for i in range(0, num_data):
            x = years_experience[i]
            y = salary[i]
            grad = model.grad_obj_func(x, y)
            upd_dir = -1.0 * grad
            update = upd_dist * upd_dir
            model.update_params(delta_w=update[0], delta_b=update[1])
        ## mse calculation
        se_sum = 0.0
        for i in range(0, num_data):
            x = years_experience[i]
            y = salary[i]
            se = model.obj_func_SE(x, y)
            se_sum += se
        se_avg = se_sum / num_data
        ## gradient calculation
        grad_sum = np.zeros(shape=[2])
        grad_avg = np.zeros(shape=[2])
        for i in range(0, num_data):
            x = years_experience[i]
            y = salary[i]
            grad = model.grad_obj_func(x, y)
            grad_sum += grad
        grad_avg = grad_sum / num_data
        l2_grad = np.linalg.norm(grad_avg, ord=2)
        print("epoch: %d, MSE: %.3f, grad:(%.2f, %.2f), L2:%.2f" % (e, se_avg, grad_avg[0], grad_avg[1], l2_grad))
    model.print_param()

    # 그래프 생성
    plt.figure(figsize=(8, 6))
    plt.scatter(years_experience, salary, label='real', color='blue')
    prediction = model.w * years_experience + model.b
    plt.plot(years_experience, prediction, label='regression', color='red')

    # 그래프 제목 및 축 레이블 추가
    plt.title('Salary vs. Years of Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    # 그리드 추가
    plt.grid(True)

    # 그래프 출력
    plt.show()

if __name__ == "__main__":
    main()