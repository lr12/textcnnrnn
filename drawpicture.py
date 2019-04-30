import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def bar():
      df=pd.DataFrame({'DT':[83,79,79],'bayes':[79,74,74],'KNN':[89,86,87],'SVM':[82,73,74],'CNN':[97,97,97],'RNN':[95,95,95],'CNN-RNN':[97,97,97]},
                      columns=['DT','bayes','KNN','SVM','CNN' ,'RNN','CNN-RNN'],
                      index=['Precision','Recall','F1-score'])


      df_plot=df.plot(kind='bar',rot=-45)
#      df_plot.legend(df.,loc=2)

      plt.ylim(ymax=100, ymin=0)

      #df_plot.set_yticks([x for x in range(1,180,10)])
    #  df_plot.set_ylabel('Longdian')
      plt.savefig('test.jpg')
      plt.show()



name_list = ['lambda=0', 'lambda=0.05', 'lambda=0.1', 'lambda=0.5']
num_list = [99, 99, 99, 90]
rects=plt.bar(range(len(num_list)), num_list, color='rgby')
# X轴标题
index=['裁定移送其他法院','准予撤诉','维持','调解']
index=[float(c)+0.4 for c in index]
plt.ylim(ymax=100, ymin=0)
plt.xticks(index, name_list)
plt.ylabel("weightedF") #X轴标签
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height)+'%', ha='center', va='bottom')

bar()