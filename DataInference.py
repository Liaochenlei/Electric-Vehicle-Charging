import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


def StructurePrediction():
    """energy mix"""
    data = r'data\PowerStructure_real.xlsx'

    data = pd.read_excel(data)
    year_real = np.arange(2018, 2024).reshape((-1, 1))
    year_pred = np.arange(2024, 2036).reshape((-1, 1))

    power_total = np.array(data.loc[:, 'Total (10^8 Kwh)'])
    model = LinearRegression()
    model.fit(year_real, power_total)
    power_total = model.predict(year_pred)
    power_real = np.array(data.loc[:, ['Hydro (10^8 Kwh)', 'Thermal (10^8 Kwh)', 'Nuclear (10^8 Kwh)',
                                       'Wind (10^8 Kwh)', 'Solar (10^8 Kwh)']])
    power_list = np.zeros((year_pred.shape[0], power_real.shape[1]))
    for i in range(power_real.shape[1]):
        model.fit(year_real, power_real[:, i])
        power_list[:, i] = model.predict(year_pred)
    # 按照总量处理
    power_list = np.round(power_list * np.tile((power_total / np.sum(power_list, axis=1)).reshape(-1, 1),
                                               power_real.shape[1]), 1)
    # rate = power_list / np.tile(np.sum(power_list, axis=1).reshape(-1, 1), power_real.shape[1])

    emission_thermal = np.array(data.loc[:, 'ThermalEmission (g/Kwh)'])
    model.fit(year_real, emission_thermal)
    emission_thermal = model.predict(year_pred)
    emission_total = np.round(emission_thermal * power_list[:, 1] / np.sum(power_list, axis=1), 1)

    line_loss = np.array(data.loc[:, 'LineLoss (%)'])
    baseline = 1 / 2
    line_loss = np.round(line_loss[0] -
                         (line_loss[0] - line_loss[-1]) / np.power(np.max(year_real) - np.min(year_real), baseline) *
                         np.power(year_pred - np.min(year_real), baseline), 2)

    data_out = np.hstack((year_pred, emission_thermal.reshape(-1, 1), emission_total.reshape(-1, 1), power_list,
                          power_total.reshape(-1, 1), line_loss.reshape(-1, 1)))
    data = pd.DataFrame(np.vstack((np.array(data), data_out)),
                        columns=['Year', 'ThermalEmission (g/Kwh)', 'TotalEmission (g/Kwh)', 'Hydro (10^8 Kwh)',
                                 'Thermal (10^8 Kwh)', 'Nuclear (10^8 Kwh)', 'Wind (10^8 Kwh)', 'Solar (10^8 Kwh)',
                                 'Total (10^8 Kwh)', 'LineLoss (%)'])
    data.to_excel('PowerStructure.xlsx', index=False)


def GDPInference():
    data_gdp = r'data\GDP.xlsx'

    data_gdp = pd.read_excel(data_gdp)
    year = np.array(data_gdp.loc[:, 'Year'])
    gdp_ppp = np.array(data_gdp.loc[:, 'GDP in PPP (billions of international dollars)'])
    index = np.argsort(year)
    year, gdp_ppp = year[index], gdp_ppp[index]

    # ARIMA
    model = ARIMA(gdp_ppp, order=(0, 2, 1))
    model_fit = model.fit()
    gdp_ppp = np.hstack((gdp_ppp, np.round(model_fit.forecast(5), 2)))

    year = np.array(data_gdp.loc[:, 'Year'])
    gdp_shanghai = np.array(data_gdp.loc[:, 'GDP of Shanghai (10^8 yuan)'])
    gdp_china = np.array(data_gdp.loc[:, 'GDP of China (10^8 yuan)'])
    index = ~np.isnan(gdp_shanghai)
    year, gdp_shanghai, gdp_china = year[index], gdp_shanghai[index], gdp_china[index]
    index = np.argsort(year)
    year, gdp_shanghai, gdp_china = year[index], gdp_shanghai[index], gdp_china[index]

    model = ARIMA(gdp_shanghai, order=(0, 2, 2))
    model_fit = model.fit()
    gdp_shanghai = np.hstack((gdp_shanghai, np.round(model_fit.forecast(12), 2)))

    model = ARIMA(gdp_china, order=(0, 2, 2))
    model_fit = model.fit()
    gdp_china = np.hstack((gdp_china, np.round(model_fit.forecast(12), 2)))

    # plt.plot(np.arange(2000, 2036), gdp_shanghai)
    # plt.plot(np.arange(2000, 2036), gdp_china)
    # plt.show()
    # plt.plot(np.arange(2000, 2036), gdp_shanghai / gdp_china)
    # plt.show()

    data = pd.DataFrame(np.vstack((np.arange(2000, 2036),
                                   gdp_ppp, np.round(gdp_shanghai / gdp_china * 100, 4),
                                   np.round(gdp_ppp * gdp_shanghai / gdp_china, 4))).T,
                        columns=['year', 'GDP of China in PPP (billions of international dollars)', 'Proportion (%)',
                                 'GDP of Shanghai in PPP (billions of international dollars)'])
    data.to_excel(r'GDP.xlsx', index=False)


def EVInference():
    data_gdp = r'data_output\GDP.xlsx'
    data_population = r'data\ShanghaiPopulationCountedbyDistrict2010to2035.xlsx'
    data_EV = r'data\EVnumber.xlsx'

    data_gdp, data_population, data_EV = pd.read_excel(data_gdp), pd.read_excel(data_population), pd.read_excel(data_EV)

    year, number = np.array(data_EV.iloc[:, 0]), np.array(data_EV.iloc[:, 1])
    index = np.argsort(year)
    year, number = year[index], number[index]

    year_all = np.arange(year[0], 2036)
    gdp, population = np.zeros(len(year_all)), np.zeros(len(year_all))
    for i in range(len(year_all)):
        gdp[i] = data_gdp.loc[data_gdp.loc[:, 'year'] == year_all[i],
                              'GDP of Shanghai in PPP (billions of international dollars)'].iloc[0]
        population[i] = data_population.loc[:, str(year_all[i])].iloc[-1]


    gdp_speed = gdp[1:] - gdp[:-1]
    population_speed = population[1:] - population[:-1]

    x = np.vstack((gdp_speed[:len(number) - 1], population_speed[:len(number) - 1])).T
    y = (number[1:] - number[:-1]).T
    model = LinearRegression()
    model.fit(x, y)
    b = model.intercept_
    a = model.coef_
    score = model.score(x, y)
    # print(a, b, score)

    prediction = np.around(a[0] * gdp_speed + a[1] * population_speed + b, 2)
    plt.scatter(year[1:], number[1:] - number[:-1], alpha=0.5)
    plt.bar(year_all[1:], prediction, alpha=0.5)
    plt.show()

    for i in range(len(number) - 1, len(year_all) - 1):
        number = np.hstack((number, number[-1] + prediction[i]))

    plt.bar(year_all, number)
    plt.show()

    prediction[:len(number) - 1] = number[1:] - number[:-1]
    data = pd.DataFrame(np.vstack((year_all, np.hstack((np.nan, prediction)), number)).T,
                        columns=['year', 'growth', 'total'])
    data.to_excel('EVnumber.xlsx', index=False)


def StationInference():

    data_station = r'data\ChargingStation.xlsx'

    data_station = pd.read_excel(data_station)

    year, number = np.array(data_station.iloc[:, 0]), np.array(data_station.iloc[:, 1])
    index = np.argsort(year)
    year, number = year[index], number[index]

    x = year.reshape((-1, 1))
    y = number.reshape((-1, 1))
    model = LinearRegression()
    model.fit(x, y)
    b = model.intercept_
    a = model.coef_
    score = model.score(x, y)
    # print(a, b, score)

    year_list = np.arange(year[0], 2036)
    prediction = np.around(a[0] * year_list + b, 0).astype(int)

    plt.scatter(year, number, alpha=0.5)
    plt.plot(year_list, prediction, alpha=0.5)
    plt.show()

    prediction[:len(year)] = number
    # 存储合并
    data = pd.DataFrame(np.vstack((year_list, prediction)).T, columns=['year', 'number'])
    data.to_excel('ChargingStationNumber.xlsx', index=False)


if __name__ == '__main__':
    EVInference()
