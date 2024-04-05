############## Installation
## https://docs.streamlit.io/get-started/installation/anaconda-distribution

import streamlit as st              # to run streamlit commands
import pandas as pd                 # for data manipulation
import numpy as np                  # for data manipulation
import altair as alt
import pyodbc                       # to read your database
import matplotlib.pyplot as plt     # to plot
import plotly.express as px         # to plot interactive plots

# Note: If you need any packages, install using the terminal (exit streamlit first):
# Keywords: pip install PackageName
# Example: pip install plotly_express

# For more charts, see:
# https://docs.streamlit.io/library/api-reference/charts
# https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart
# https://matplotlib.org/stable/plot_types/index.html
# https://altair-viz.github.io/gallery/index.html
# https://docs.streamlit.io/library/api-reference/charts/st.altair_chart
####################################################

# Page configuration
st.set_page_config(
    page_title="Group X - Final Project",
    layout="wide",
    initial_sidebar_state="expanded")
alt.themes.enable("dark")

# Connect to SQL Server. Prepare your secret file before running this code.
# The example below uses Mr Plan's login details (some details are changed)
#######################
# server = "COMPUTERNAME\SQLEXPRESS"
# database = "Vehicle"
# username = "sa"
# password = "sqlserver123"
# sqlcmd -S localhost -U username -P
#######################
st.cache_resource.clear()
@st.cache_resource
def init_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER="+ st.secrets["server"]
        + ";DATABASE="+ st.secrets["database"]
        + ";UID="+ st.secrets["username"]
        + ";PWD="+ st.secrets["password"]
    )
st.write("This is the connection string (check if details are correct, then delete when working:")
st.write("DRIVER={ODBC Driver 17 for SQL Server};SERVER="+ st.secrets["server"]
        + ";DATABASE="+ st.secrets["database"]
        + ";UID="+ st.secrets["username"]
        + ";PWD="+ st.secrets["password"])
conn = init_connection()

st.sidebar.title('Group X')
options = st.sidebar.radio('Pages',
                            options=['Sample Charts','Database Charts','Data'])

# Functions to be used 
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)

# Just runs a query. Check "Data page" to see how it works
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

# Runs a query and saves the data into a dataframe.
    #currently inefficient but it works for small datasets.
    #Note this function must be redefined to adjust to the number of columns automatically.
def run_query_df(query):
    data=conn.execute(query)
    df=pd.DataFrame.from_dict(data.fetchall())
    #st.write(df.columns)
    [nrows,ncols]=df.shape
    actualcols=2
    df2=pd.DataFrame(index=np.arange(nrows), columns=np.arange(actualcols))
    for i in np.arange(0,nrows):
        strfull=df.iloc[i,0]
        for j in np.arange(0,actualcols):
            df2.iloc[i,j]=strfull[j]
    return df2


##################### Page details

######################### Sample Charts Page
if options =="Sample Charts":
    st.write('Show sample charts here using the streamlit command (and random data generated.)')
    a=np.random.randn(1,3)
    col = st.columns((1, 3, 3), gap='medium')
    with col[0]:
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        st.bar_chart(chart_data)
    with col[1]:
        chart_data2 = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        st.line_chart(chart_data2)
    with col[2]:
        chart_data3 = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        st.line_chart(chart_data3)

############################### Database Charts Page
elif options =="Database Charts":
    st.write('Sample chart using matplotlib')
    col = st.columns((3, 3, 3), gap='medium')
    with col[0]:
        st.write("Let us use a dataframe.")
        df_cars=run_query_df("SELECT VehicleBrand, AVG(ProdHours) FROM Vehicle WHERE Vstatus=1 GROUP BY VehicleBrand ORDER BY AVG(Prodhours) DESC")
        df_cars.columns=['Brand','AvePrice']
        st.write(df_cars)
        #st.pyplot(df_cars.plot.barh().figure)
        fig=px.bar(df_cars, x='Brand', y='AvePrice', width=400)
        st.plotly_chart(fig)
    with col[1]:
    # Some random chart using matplotlib histogram
        arr = np.random.normal(1, 1, size=100)
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        st.pyplot(fig)
    with col[2]:
        rows3= run_query("SELECT VehicleBrand, AVG(ProdHours) FROM Vehicle WHERE Vstatus=1 GROUP BY VehicleBrand ORDER BY AVG(Prodhours) DESC")
        vbrand=[]
        avghours=[]
        for row in rows3:
            vbrand.append(row[0])
            avghours.append(row[1])

        fig, ax= plt.subplots()
        ax.bar(vbrand,avghours,width=0.5,color=['r','g'])
        with col[2]:
            st.pyplot(fig)
        
################################### Data Page
elif options == 'Data':
    st.write('Show tables here')
    #Example 1
    st.write("Example 1")
    rows = run_query("SELECT * from Employee;")
    for row in rows:
        st.write(f"{row[0]} has a :{row[1]}:--{row[2]}:--{row[3]}:--{row[4]}")

    #Example 2
    st.write("Example 2")
    rows2 = run_query("SELECT VehicleID,VStatus,ProdHours from Vehicle WHERE Vstatus=1;")
    for row in rows2:
        st.write(f"{row[0]} has a :{row[1]}:--{row[2]}")

    #Example 3
    st.write("Example 3")
    df3 = run_query_df("SELECT VehicleID,VStatus,ProdHours from Vehicle WHERE Vstatus=1;")
    st.write(df3)
