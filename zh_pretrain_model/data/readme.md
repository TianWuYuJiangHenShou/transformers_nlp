任务1：OCNLI–中文原版自然语言推理
0 一月份跟二月份肯定有一个月份有. 肯定有一个月份有 0  
1 一月份跟二月份肯定有一个月份有. 一月份有 1  
2 一月份跟二月份肯定有一个月份有. 一月二月都没有 2  
3 一点来钟时,张永红却来了 一点多钟,张永红来了 0  
4 不讲社会效果,信口开河,对任何事情都随意发议论,甚至信谣传谣,以讹传讹,那是会涣散队伍、贻误事业的 以讹传讹是有害的 0  
（注：id 句子1 句子2 标签）
（注：标签集合：[蕴含，中性，不相关]）

任务2：OCEMOTION–中文情感分类
0 你知道多伦多附近有什么吗?哈哈有破布耶...真的书上写的你听哦...你家那块破布是世界上最大的破布,哈哈,骗你的啦它是说尼加拉瓜瀑布是世界上最大的瀑布啦...哈哈哈''爸爸,她的头发耶!我们大扫除椅子都要翻上来我看到木头缝里有头发...一定是xx以前夹到的,你说是不是?[生病] sadness  
1 平安夜,圣诞节,都过了,我很难过,和妈妈吵了两天,以死相逼才终止战争,现在还处于冷战中。sadness  
2 我只是自私了一点,做自己想做的事情! sadness  
3 让感动的不仅仅是雨过天晴,还有泪水流下来的迷人眼神。happiness  
4 好日子 happiness  
（注：id 句子 标签）
len(label) = 7

任务3：TNEWS–今日头条新闻标题分类
0 上课时学生手机响个不停,老师一怒之下把手机摔了,家长拿发票让老师赔,大家怎么看待这种事? 108  
1 商赢环球股份有限公司关于延期回复上海证券交易所对公司2017年年度报告的事后审核问询函的公告 104  
2 通过中介公司买了二手房,首付都付了,现在卖家不想卖了。怎么处理? 106  
3 2018年去俄罗斯看世界杯得花多少钱? 112  
4 剃须刀的个性革新,雷明登天猫定制版新品首发 109  
（注：id 句子 标签）
len(label) = 15


