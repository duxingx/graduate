library(readr)
library(tidyverse)
library(Hmisc) # 相关系数
library(glmnet) # 弹性网变量选择
library(broom) # 模型结果整理
library(car) # qq图
# library(MASS) # 岭回归
##############################################################################
# 计算波峰波谷
findpeak_boo <- function(x){
  r <- rle(x)
  pks <- which(rep(x = diff(sign(diff(c(-Inf, r$values, -Inf)))) == -2, 
                   times = r$lengths))
  boos <- which(rep(x = diff(sign(diff(c(-Inf, r$values, -Inf)))) == 2, 
                    times = r$lengths))
  n_pks <- length(pks)
  n_boos <- length(boos)
  return(list(
    "pks"=pks,
    "boos"=boos,
    "n_pks"=n_pks,
    "n_boos"=n_boos
  ))
}
# 缺失值均值填补
fillna = function(x){
  x[is.na(x)] = mean(x,na.rm = TRUE)
  return(x)
}
##############################################################################


# 数据导入
################################################################################
setwd("D:\\OneDrive\\A_Study\\A5_FinaBigData\\A4_HomeWorkPaper")
rawdata1 = readr::read_csv("./data/Appendix1.csv")
rawdata2 = readr::read_csv("./data/Appendix2.csv",skip = 1,col_names = FALSE)

# 数据预处理
data1_raw = rawdata1 %>% as_tibble() %>% drop_na()
data2_raw = rawdata2 %>% mutate(Time=(1:nrow(.))) %>%        # 添加采样时间
            map_dfc(~ replace_na(.x, rstatix::get_mode(.x)[1])) %>% # 缺失值填补
            as_tibble() 
data1.scale = data1_raw %>% 
  mutate_at(vars(-Time),~ scale(., center = T, scale = T))  # 数据标准化

data2.scale = data2_raw %>% 
  mutate_at(vars(-Time),~ scale(., center = T, scale = T)) # 数据标准化



# 合并X、Y数据
data_all = data1_raw %>% 
  dplyr::select(Time,PipeMean) %>% 
  left_join(data2.scale,by="Time")

# 异常值检测
mod <- lm(PipeMean ~ ., data=dplyr::select(data_all,-Time))
cooksd <- cooks.distance(mod)
influences = which(cooksd > 4*mean(cooksd, na.rm=T))
data1_raw[influences,]
# plot(cooksd, pch="*", cex=1, main="Influential Obs by Cooks distance")  # 绘制Cook距离
# abline(h = 4*mean(cooksd, na.rm=T), col="red")  # 添加决策线
# text(x=1:length(cooksd)+1, y=cooksd, 
#      labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), 
#      col="red")  # 添加标签

# 删除异常值
data1_raw = data1_raw[-influences,]
data2_raw = data2_raw[-influences,]
data1.scale = data1.scale[-influences,]
data2.scale = data2.scale[-influences,]
data_all = data_all[-influences,]


# 宽数据转长数据
data1_longer = data1_raw %>% 
  pivot_longer(cols = -Time,names_to="Pipes",values_to="Temperature")
data2_longer = data2_raw %>% 
  pivot_longer(cols = -Time,names_to="Variables",values_to="Value")

################################################################################

## 因变量的统计特征
################################################################################
Y.res_describe = data1_longer %>%
  group_by(Pipes) %>%
  summarise(
    Mean = round(mean(Temperature),2),
    Std = round(sd(Temperature),2),
    Max = round(max(Temperature),2),
    Min = round(min(Temperature),2),
    Q25 = round(quantile(Temperature,.75),2),
    Q50 = round(median(Temperature),2),
    Q75 = round(quantile(Temperature,.25),2),
    PeakNumber = round(findpeak_boo(Temperature)[["n_pks"]],2),
    TroughNumber = round(findpeak_boo(Temperature)[["n_boos"]],2)
  )
Y.res_describe
################################################################################

# 自变量的统计特征
################################################################################
X.res_describe = data2_longer %>%
  group_by(Variables) %>%
  summarise(
    Mean = round(mean(Value),2),
    Std = round(sd(Value),2),
    Max = round(max(Value),2),
    Min = round(min(Value),2),
    Q25 = round(quantile(Value,.75),2),
    Q50 = round(median(Value),2),
    Q75 = round(quantile(Value,.25),2),
    PeakNumber = round(findpeak_boo(Value)[["n_pks"]],2),
    TroughNumber = round(findpeak_boo(Value)[["n_boos"]],2)
  )
X.res_describe
################################################################################

# 应变量的箱线图
################################################################################
res_boxplot = data1_longer %>%
  ggplot(aes(x=Pipes,y=Temperature)) + 
  geom_boxplot(aes(group=Pipes,color=Pipes,fill=Pipes))+
  ggthemes::theme_few()+
  ggtitle("Plot 1.  Boxing diagram of pipe temperature")
res_boxplot
################################################################################

# 管道温度的时序图
################################################################################
res_lineplot = data1_longer %>%
  ggplot(aes(x=Time,y=Temperature)) + 
  geom_line(aes(group=Pipes,color=Pipes)) +
  ggtitle("Plot2. Time sequence diagram of the temperature of each pipe") + 
  labs(x="Sampling time point",y="Pipe temperature") +
  ggthemes::theme_few()+
  facet_grid(Pipes~.) 
res_lineplot
################################################################################

# 因变量与自变量的相关系数
################################################################################
# 计算相关系数
corr = rcorr(as.matrix(data1_raw),as.matrix(dplyr::select(data2_raw,-Time)))
corr2 = as_tibble(corr[[1]]) %>% dplyr::select(-c(names(data1_raw)))
cor = corr2[1:11,] %>% pivot_longer(cols=names(corr2),names_to="Variables",values_to="Correlation")

# 相关系数的箱线图
res3_boxplot = cor %>%
  ggplot(aes(x=fct_reorder(Variables, Correlation, .fun = median),y=Correlation)) + 
  geom_boxplot(aes(group=Variables,
                   color=fct_reorder(Variables, Correlation, .fun = median),
                   fill=fct_reorder(Variables, Correlation, .fun = median))) + 
  geom_hline(yintercept = c(0.5,-0.5),linetype=2)+
  ggtitle("Plot3. Box plot of correlation coefficient between independent variable and pipe temperature") + 
  theme(
    plot.title = element_text(hjust=0.5,color="black",size=14),
    plot.margin = margin(t=4,r=3,b=4,l=3,unit='pt'),
    axis.line = element_line(size=1),
    axis.title = element_text(size=14),
    axis.text.x = element_text(angle = 90, hjust = 1,size = 7),
    axis.title.x = element_text(size = 14, vjust = 0.5, hjust = 0.5),
    legend.position="none")
res3_boxplot
################################################################################

# 管道温度均值的时序图
################################################################################
res_lineplot = data1_longer %>%
  filter(Pipes == "PipeMean") %>%
  ggplot(aes(x=Time,y=Temperature)) + 
  geom_line() +
  ggtitle("Plot4. Time sequence diagram of the mean temperature of all pipe") + 
  labs(x="Sampling time point",y="Pipe mean temperature") +
  ggthemes::theme_few()
res_lineplot
################################################################################

# 数据集合并,划分训练集和测试集（时序前后70%、30%）、标准化
################################################################################

# 划分数据集
train_data = filter(data_all, Time <= nrow(data_all)*0.7)
test_data = filter(data_all, Time > nrow(data_all)*0.7)
dim(train_data)
X.train = as.matrix(dplyr::select(train_data,-Time,-PipeMean))
X.test = as.matrix(dplyr::select(test_data,-Time,-PipeMean))
Y.train = as.matrix(dplyr::select(train_data,PipeMean))
Y.test = as.matrix(dplyr::select(test_data,PipeMean))
################################################################################

# 变量选择
################################################################################
# 交叉验证收敛图
op = par(no.readonly = TRUE)
par(mfrow=c(2,2))
for(i in c(0,0.25,0.75,1)){
  cvfit = cv.glmnet(X.train,Y.train,alpha=i)
  # alpha=0:Ridge
  # alpha=1:Lasso
  # alpha=0.25,0.78:弹性网
  plot(cvfit)
  print(cvfit$lambda.min)
}
par(op)

# 变量选择结果
cvfit4 = cv.glmnet(X.train,Y.train,alpha=1)
z = coef.glmnet(cvfit,s=cvfit4$lambda.1se)
var_index = z@i[-1] # 提取筛选变量的索引

# 筛选后的数据集
X.train_dplyr::select = X.train[,z@i[-1]]
X.test_dplyr::select = X.test[,z@i[-1]]

train = as_tibble(X.train_dplyr::select)
test = as_tibble(X.test_dplyr::select)

train["PipeMean"] = train_data$PipeMean
test["PipeMean"] = test_data$PipeMean
################################################################################

# 多元线性模型
################################################################################
ols = lm(PipeMean~.,data = train)

# 模型检验
glance(ols)

# 系数结果
ols %>% tidy(conf.int = TRUE)
# 系数结果可视化
ols %>% tidy() %>% filter(term != "(Intercept)") %>%
  ggplot(mapping = aes(x = term,y = estimate)) +
  geom_point() + 
  coord_flip()
# 系数可视化（有置信区间）
ols %>% tidy(conf.int = TRUE) %>%
  filter(!term %in% c("(Intercept)")) %>%
  ggplot(aes(x = reorder(term, estimate),y = estimate,
             ymin = conf.low, ymax = conf.high)) +
  geom_pointrange() +
  coord_flip() + labs(x = "", y = "OLS Estimate")
################################################################################

# OLS普通残差
################################################################################
r <- residuals(ols)   
# 残差图
op = par(no.readonly = T)
par(mar = c(6, 4, 0.5, 0.5), mfrow = c(2, 1))
y.fit <- predict(ols)       # 预测值
y.rst1 <- rstandard(ols)    # 内学生化残差
y.rst2 <- rstudent(ols)     # 外学生化残差
plot(1:length(y.rst1),y.rst1,type = "l",pch = 19,cex = 0.5,
     sub = "图7-1(a) 内学生化残差图")    
abline(h = 0,col = "red",lwd = 2)
plot(1:length(y.rst2),y.rst2,type = "l",pch = 19,cex = 0.5,
     sub = "图7-1(b) 外学生化残差图")
abline(h = 0,col = "red",lwd = 2)
par(mfrow=c(1,1))

# 残差的W正态性检验
shapiro.test(r) %>% tidy()
# 对数正态QQ残差图
qqPlot(ols, col="blue", col.lines="red",sub = "图7-2 正态QQ残差图")
################################################################################

# OLS模型拟合、预测
################################################################################
train_res = train %>% 
  modelr::add_predictions(ols) %>%
  modelr::add_residuals(ols)
test_res = test %>% 
  modelr::add_predictions(ols) %>%
  modelr::add_residuals(ols)
all = c(train_res$PipeMean,test_res$PipeMean)

# 预测图
plot(1:length(all),all,type="l",col="black",xlab="时间",ylab = "上证50指数",
     lty=1,cex=0.8,pch=16,sub = "图6  指数追踪与预测图")
points(1:length(train$PipeMean),train_res$pred,type="l",col="blue",lty=2,cex=0.8,pch=0)
points((1:length(all))[-(1:length(train$PipeMean))],test_res$pred,type="l", 
       col="red",lty=2,cex=0.8,pch=2)
abline(v=max(1:length(train$PipeMean)),lty=2)
legend("topleft",c("真实值","拟合值","预测值"),col=c('black','blue','red'),
       lty=c(1,2,2),pch=c(16,0,2),bty='n')

# 模型效果
MSE.train = mean(train_res$resid)
SSE.train = sum(train_res$resid)
MSE.test = mean(test_res$resid)
SSE.test = sum(test_res$resid)
################################################################################

# 岭回归模型
################################################################################
cv=cv.glmnet(X.train,Y.train,alpha=0)
model.elast = glmnet(X.train,Y.train,lambda = cv$lambda.min)

glance(model.elast)
model.elast %>% tidy(conf.int = TRUE)

z= coef(model.elast)  #系数
################################################################################

# 岭回归模型拟合、预测

pred.train = predict.glmnet(model.elast,X.train)
pred.test = predict.glmnet(model.elast,X.test)

all = c(Y.train,Y.test)

plot(1:length(all),all,type="l",col="black",xlab="时间",ylab = "上证50指数",lty=1,cex=0.8,pch=16,sub = "图6  指数追踪与预测图")
points(1:length(Y.train),pred.train,type="l",col="blue",lty=2,cex=0.8,pch=0)
points((1:length(all))[-(1:length(Y.train))],pred.test,type="l", col="red",lty=2,cex=0.8,pch=2)
abline(v=max(1:round(length(all)*0.7)),lty=2)
legend("topleft",c("真实值","拟合值","预测值"),col=c('black','blue','red'),lty=c(1,2,2),pch=c(16,0,2),bty='n')

MSE.train = mean(Y.train-pred.train)
SSE.train = sum(Y.train-pred.train)
MSE.test = mean(Y.test-pred.test)
SSE.test = sum(Y.test-pred.test)
