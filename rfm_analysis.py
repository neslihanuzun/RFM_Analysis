import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv(data_20K.csv")
df = df_.copy()

df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.info()


#Adım 3  Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
#master_id Eşsiz #müşteri numarası
#order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
#last_order_channel En son alışverişin yapıldığı kanal
#first_order_date Müşterinin yaptığı ilk alışveriş tarihi
#last_order_date Müşterinin yaptığı son alışveriş tarihi
#last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
#last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
#order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
#order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
#customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
#customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
#interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

df["omnichannel_order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["omnichannel_customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


df.head()
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)
df['first_order_date'] = df['first_order_date'].astype('datetime64[ns]')
df['last_order_date'] = df['last_order_date'].astype('datetime64[ns]')
df['last_order_date_online'] = df['last_order_date_online'].astype('datetime64[ns]')
df['last_order_date_offline'] = df['last_order_date_offline'].astype('datetime64[ns]')


df.groupby('order_channel').agg({"master_id": lambda master_id : master_id.nunique(),
                                 "omnichannel_order_num_total": "mean",
                                 "omnichannel_customer_value_total": "mean"})

#Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df["omnichannel_customer_value_total"].sort_values(ascending=False).head(10)


df["omnichannel_order_num_total"].sort_values(ascending=False).head(10)


df.shape
df.isnull().sum()
df.describe([0.25, 0.5, 0.75]).T


df.head()
#??df[today_date] - df["last_order_date"].max()
df["omnichannel_order_num_total"].nunique()
df["omnichannel_customer_value_total"].sum()


# Recency müşterinin kaç gun once sipariş verdiğini,
# Frequency müşterinin kaç sipariş verdiğini,
# Monetary ise müşteriden elde edilen toplam geliri ifade etmektedir.

df.groupby("master_id").agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                            'omnichannel_order_num_total': lambda omnichannel_order_num_total: omnichannel_order_num_total.nunique(),
                            'omnichannel_customer_value_total': lambda omnichannel_customer_value_total: omnichannel_customer_value_total.sum()})


rfm = df.groupby("master_id").agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'omnichannel_order_num_total': lambda omnichannel_order_num_total: omnichannel_order_num_total.nunique(),
                                     'omnichannel_customer_value_total': lambda omnichannel_customer_value_total: omnichannel_customer_value_total.sum()})
rfm.head()


rfm.columns = ['recency', 'frequency', 'monetary']

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm[rfm["RF_SCORE"] == "55"]

rfm[rfm["RF_SCORE"] == "11"]

#hibernating, kaybediilen müşteri
#at_Risk, kaybedilme riski olan müşteri,
#cant_loose, kaybetmemek için önlem alınması gereken müşteri,
#about_to_sleep, durağanlaşmış müşteri,
#need_attention, dikkat edilmesi gereken müşteri
#loyal_customers, önemli müşteri
#promising, tutundurulması gereken müşteri
#new_customers, yeni müşteriler
#potential_loyalists, potansıyel onemli müşteri,
#champions hem yenı alısverıs yapmıs hem frekansı yuksek olan musterıler

#Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)


rfm[["recency", "frequency", "monetary"]].groupby([rfm['segment']]).agg(["mean"])


women = df[(df['interested_in_categories_12']).str.contains("KADIN")]
high_value_customers = rfm[(rfm["segment"].isin(["champions","loyal_customers"]))]
w_hvc = pd.merge(women, high_value_customers, on=["master_id"])
w_hvc.head()


men_child = df[(df['interested_in_categories_12']).str.contains("ERKEK", "COCUK")]
attention_customers = rfm[(rfm["segment"].isin(['cant_loose', 'about_to_sleep', 'new_customers']))]
mc_ac = pd.merge(men_child, attention_customers, on=["master_id"])
