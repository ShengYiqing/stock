select t.* from
(
select t.*, rank() over(order by mc desc)lead_stock from
(
select t.*, rank() over(partition by l1_name, l2_name, l3_name order by mc desc) lead_ind from
(
select tl.trade_date, tl.stock_code, tl.price, tsb.name, ti.l1_name, ti.l2_name, ti.l3_name, ts.mc, ts.rank_mc, ts.pb, ts.rank_pb, ts.pe, ts.rank_pe, ts.roe, ts.rank_roe, ts.beta, ts.rank_beta
from label.tdailylabel tl
left join tsdata.ttsstockbasic tsb
on tl.stock_code = tsb.stock_code
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
left join style.tdailystyle ts
on tl.trade_date = ts.trade_date
and tl.stock_code = ts.stock_code
where tl.trade_date = '20231012'
and tl.amount >= 100000 and tl.price >= 5
and ts.rank_mc >= 0.8 and ts.rank_pb >= 0.2 and ts.rank_pe >= 0.05 and ts.rank_roe >= 0.05
and ti.l3_name in (
'快递', 
'航空运输', 
'大众出版', '广告媒体', '影视动漫制作', '文字媒体', '视频媒体', '游戏Ⅲ', 
'多业态零售', '百货', '超市', '旅游零售Ⅲ', 
'其他家电Ⅲ', '卫浴电器', '厨房电器', '个护小家电', '厨房小家电', '清洁小家电', '冰洗', '空调', '其他黑色家电', '彩电', 
'电动乘用车', '综合乘用车', '其他运输设备', '摩托车', 
'品牌消费电子', 
'体育Ⅲ', '人工景区', '旅游综合', '酒店', '餐饮', 
'家纺', '运动服装', '非运动服装', '鞋帽及其他', '纺织鞋类制造', '其他饰品', '钟表珠宝', 
'洗护用品', '生活用纸', '品牌化妆品', '医美服务', '医美耗材', 
'娱乐用品', '文化用品', 
'电信运营商', 
'烘焙食品', '熟食', '零食', '白酒Ⅲ', '调味发酵品Ⅲ', '其他酒类', '啤酒', '保健品', '肉制品', '预加工食品', '乳品', '软饮料')
) t
) t
where t.lead_ind <= 5
) t
where t.lead_stock <= 50
order by l1_name, l2_name, l3_name, lead_stock