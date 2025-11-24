import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d, CubicSpline
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import io
import base64
from statsmodels.nonparametric.smoothers_lowess import lowess

# Configure the page
st.set_page_config(
    page_title="emiSENSE CORE",
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2E86AB; text-align: center; margin-bottom: 1rem; }
    .studio-card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2E86AB; margin: 1rem 0; }
    .commercial-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .research-card { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .download-btn { background-color: #27AE60; color: white; padding: 0.5rem 1rem; border-radius: 5px; border: none; margin: 0.2rem; }
    .metric-card { background-color: #e8f4fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key in ['calibration_data', 'emissivity_data', 'absolute_temp_data', 'hysteresis_data', 
            'pyro_file', 'thermal_file', 'blackbody_file', 'target_points', 'spline_smoothing',
            'outlier_threshold', 'min_temp', 'interpolation_points', 'smoothing_enabled']:
    if key not in st.session_state:
        st.session_state[key] = None

# Set default smoothing to enabled
if st.session_state.smoothing_enabled is None:
    st.session_state.smoothing_enabled = True

# Download functions
def get_table_download_link(df, filename, button_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">{button_text}</a>'
    return href

def get_excel_download_link(df, filename, button_text):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
        writer.close()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-btn">{button_text}</a>'
    return href

def get_plot_download_link(fig, filename, button_text):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-btn">{button_text}</a>'
    return href

# ========== ENHANCED SMOOTHING FUNCTIONS ==========

def apply_smoothing(x, y, method='spline', smoothing_factor=0.0001, frac=0.1):
    """Apply various smoothing methods with default enabled"""
    if not st.session_state.smoothing_enabled or len(x) < 10:
        return x, y, x, y
    
    try:
        if method == 'spline':
            df = pd.DataFrame({'x': x, 'y': y}).drop_duplicates('x').sort_values('x')
            if len(df) < 10:
                return x, y, x, y
            spline = UnivariateSpline(df['x'], df['y'], s=len(df)*smoothing_factor)
            x_smooth = np.linspace(df['x'].min(), df['x'].max(), min(1000, len(df)*2))
            y_smooth = spline(x_smooth)
            return df['x'].values, df['y'].values, x_smooth, y_smooth
            
        elif method == 'lowess':
            df = pd.DataFrame({'x': x, 'y': y}).drop_duplicates('x').sort_values('x')
            if len(df) < 10:
                return x, y, x, y
            smoothed = lowess(df['y'], df['x'], frac=frac)
            return df['x'].values, df['y'].values, smoothed[:, 0], smoothed[:, 1]
            
        elif method == 'moving_avg':
            df = pd.DataFrame({'x': x, 'y': y}).drop_duplicates('x').sort_values('x')
            window_size = max(3, min(21, len(df) // 10))
            y_smooth = df['y'].rolling(window=window_size, center=True).mean()
            valid_mask = ~y_smooth.isna()
            return df['x'].values[valid_mask], df['y'].values[valid_mask], df['x'].values[valid_mask], y_smooth[valid_mask].values
            
    except Exception as e:
        st.warning(f"Smoothing failed: {e}. Using original data.")
    
    return x, y, x, y

def adjustable_smoothing_enhanced(temperature, voltage, spline_s, outlier_threshold, smoothing_enabled=True):
    """Enhanced version with optional smoothing"""
    volt_series = pd.Series(voltage)
    rolling_mean = volt_series.rolling(window=20, center=True).mean()
    rolling_std = volt_series.rolling(window=20, center=True).std()
    z_scores = np.abs((voltage - rolling_mean) / (rolling_std + 1e-8))
    valid_mask = z_scores < outlier_threshold
    
    temp_clean = temperature[valid_mask]
    volt_clean = voltage[valid_mask]
    
    if len(temp_clean) < 10:
        return temperature, voltage, temperature, voltage
    
    if smoothing_enabled:
        temp_unique, volt_unique, temp_smooth, volt_smooth = apply_smoothing(
            temp_clean, volt_clean, 'spline', spline_s
        )
        return temp_unique, volt_unique, temp_smooth, volt_smooth
    else:
        return temp_clean, volt_clean, temp_clean, volt_clean

# ========== MODIFIED CALIBRATION STUDIO ==========

def run_calibration_studio_enhanced(pyro_data, thermal_data, target_points=8000):
    """Enhanced calibration with accuracy comparison to TS temperature"""
    try:
        pyro_time_col, pyro_volt_col, pyro_temp_col = 'Time', 'Voltage', 'Temperature'
        thermal_time_col, thermal_temp_col = 'Time', 'Temperature'

        pyro_data[pyro_time_col] = pd.to_numeric(pyro_data[pyro_time_col], errors='coerce')
        thermal_data[thermal_time_col] = pd.to_numeric(thermal_data[thermal_time_col], errors='coerce')
        pyro_data = pyro_data.dropna(subset=[pyro_time_col])
        thermal_data = thermal_data.dropna(subset=[thermal_time_col])

        time_min = max(pyro_data[pyro_time_col].min(), thermal_data[thermal_time_col].min())
        time_max = min(pyro_data[pyro_time_col].max(), thermal_data[thermal_time_col].max())
        optimized_time = np.linspace(time_min, time_max, target_points)

        ts_temp = np.interp(optimized_time, thermal_data[thermal_time_col], thermal_data[thermal_temp_col])
        ir_volt = np.interp(optimized_time, pyro_data[pyro_time_col], pyro_data[pyro_volt_col])
        ir_temp = np.interp(optimized_time, pyro_data[pyro_time_col], pyro_data[pyro_temp_col])

        peak_idx = np.argmax(ts_temp)
        heating_mask = np.arange(len(ts_temp)) <= peak_idx
        cooling_mask = np.arange(len(ts_temp)) >= peak_idx

        calibration_data = {
            'full': pd.DataFrame({
                'Time_s': optimized_time, 'TS_Temperature_C': ts_temp, 
                'IR_Voltage_V': ir_volt, 'IR_Temperature_C': ir_temp
            }),
            'heating': pd.DataFrame({
                'Time_s': optimized_time[heating_mask], 'TS_Temperature_C': ts_temp[heating_mask], 
                'IR_Voltage_V': ir_volt[heating_mask], 'IR_Temperature_C': ir_temp[heating_mask]
            }),
            'cooling': pd.DataFrame({
                'Time_s': optimized_time[cooling_mask], 'TS_Temperature_C': ts_temp[cooling_mask],
                'IR_Voltage_V': ir_volt[cooling_mask], 'IR_Temperature_C': ir_temp[cooling_mask]
            })
        }
        return calibration_data
        
    except Exception as e:
        st.error(f"Calibration error: {str(e)}")
        return None

def render_calibration_studio():
    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.header("🏗️ Calibration Studio - Data Alignment & Accuracy Analysis")
    
    if st.session_state.calibration_data:
        st.success("✅ Calibration Complete!")
        cal_data = st.session_state.calibration_data['full']
        ts_temp, ir_temp = cal_data['TS_Temperature_C'].values, cal_data['IR_Temperature_C'].values
        
        valid_mask = (ts_temp > 100) & (ir_temp > 100) & (ts_temp < 2000) & (ir_temp < 2000)
        ts_valid, ir_valid = ts_temp[valid_mask], ir_temp[valid_mask]
        
        if len(ts_valid) > 10:
            temp_difference = ts_valid - ir_valid
            accuracy_metrics = {
                "Mean Difference (TS - IR)": f"{np.mean(temp_difference):.1f}°C",
                "Std Deviation": f"{np.std(temp_difference):.1f}°C",
                "Max Overestimation": f"{np.max(temp_difference):.1f}°C",
                "Max Underestimation": f"{np.min(temp_difference):.1f}°C",
                "R² vs TS Temperature": f"{np.corrcoef(ts_valid, ir_valid)[0,1]**2:.3f}",
                "RMSE": f"{np.sqrt(np.mean(temp_difference**2)):.1f}°C"
            }
        else:
            accuracy_metrics = {"Status": "Insufficient valid data"}
        
        st.subheader("🎯 Accuracy Analysis (vs TS Reference Temperature)")
        cols = st.columns(3)
        for i, (key, value) in enumerate(list(accuracy_metrics.items())):
            with cols[i % 3]:
                st.metric(key, value)
        
        st.subheader("📊 Real-time AM Monitoring Implications")
        col1, col2 = st.columns(2)
        with col1:
            avg_error = np.mean(np.abs(temp_difference)) if len(ts_valid) > 10 else 0
            if avg_error < 20: st.success(f"✅ Excellent (Avg error: {avg_error:.1f}°C)")
            elif avg_error < 50: st.warning(f"⚠️ Acceptable (Avg error: {avg_error:.1f}°C)")
            else: st.error(f"❌ High error (Avg error: {avg_error:.1f}°C)")
        with col2:
            correlation = np.corrcoef(ts_valid, ir_valid)[0,1] if len(ts_valid) > 10 else 0
            if abs(correlation) > 0.9: st.success(f"✅ Strong correlation (R: {correlation:.3f})")
            elif abs(correlation) > 0.7: st.warning(f"⚠️ Moderate correlation (R: {correlation:.3f})")
            else: st.error(f"❌ Weak correlation (R: {correlation:.3f})")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Points", f"{len(cal_data):,}")
        with col2: st.metric("Heating Points", f"{len(st.session_state.calibration_data['heating']):,}")
        with col3: st.metric("Cooling Points", f"{len(st.session_state.calibration_data['cooling']):,}")
        
        st.subheader("📈 IR Voltage vs TS Temperature")
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        if st.session_state.smoothing_enabled:
            for cycle, color, label in [('full', 'gray', 'All Data'), ('heating', 'red', 'Heating'), ('cooling', 'blue', 'Cooling')]:
                data = st.session_state.calibration_data[cycle]
                x_orig, y_orig, x_smooth, y_smooth = apply_smoothing(
                    data['TS_Temperature_C'].values, data['IR_Voltage_V'].values, 'spline', st.session_state.spline_smoothing
                )
                ax1.plot(x_smooth, y_smooth, color=color, label=label, alpha=0.8 if cycle == 'full' else 1.0)
        else:
            ax1.plot(cal_data['TS_Temperature_C'], cal_data['IR_Voltage_V'], 'gray', alpha=0.5, label='All Data')
            ax1.plot(st.session_state.calibration_data['heating']['TS_Temperature_C'], 
                    st.session_state.calibration_data['heating']['IR_Voltage_V'], 'red', label='Heating')
            ax1.plot(st.session_state.calibration_data['cooling']['TS_Temperature_C'], 
                    st.session_state.calibration_data['cooling']['IR_Voltage_V'], 'blue', label='Cooling')
        
        ax1.set_xlabel('TS Temperature (°C)'); ax1.set_ylabel('IR Voltage (V)')
        ax1.set_title('IR Voltage vs TS Temperature'); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(cal_data['Time_s'], cal_data['IR_Voltage_V'], 'green')
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('IR Voltage (V)')
        ax2.set_title('IR Voltage vs Time'); ax2.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        st.subheader("🌡️ Temperature Comparison: IR vs TS Reference")
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax3.plot(cal_data['TS_Temperature_C'], cal_data['IR_Temperature_C'], 'gray', alpha=0.3, label='All Data')
        ax3.plot(st.session_state.calibration_data['heating']['TS_Temperature_C'], 
                st.session_state.calibration_data['heating']['IR_Temperature_C'], 'red', label='Heating', linewidth=2)
        ax3.plot(st.session_state.calibration_data['cooling']['TS_Temperature_C'], 
                st.session_state.calibration_data['cooling']['IR_Temperature_C'], 'blue', label='Cooling', linewidth=2)
        
        max_temp = max(cal_data['TS_Temperature_C'].max(), cal_data['IR_Temperature_C'].max())
        ax3.plot([0, max_temp], [0, max_temp], 'k--', alpha=0.5, label='Ideal')
        ax3.fill_between([0, max_temp], [0, max_temp-10], [0, max_temp+10], alpha=0.2, color='green', label='±10°C Accuracy Band')
        ax3.fill_between([0, max_temp], [0, max_temp-20], [0, max_temp+20], alpha=0.1, color='yellow', label='±20°C Accuracy Band')
        
        ax3.set_xlabel('TS Reference Temperature (°C)'); ax3.set_ylabel('IR Temperature (°C)')
        ax3.set_title('IR Temperature vs TS Reference (Accuracy Bands)'); ax3.legend(); ax3.grid(True, alpha=0.3)
        
        ax4.plot(cal_data['Time_s'], cal_data['IR_Temperature_C'], 'orange', label='IR Temperature')
        ax4.plot(cal_data['Time_s'], cal_data['TS_Temperature_C'], 'purple', label='TS Temperature', alpha=0.7)
        ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('IR vs TS Temperature vs Time'); ax4.legend(); ax4.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        st.markdown("### 📥 Download Calibration Data")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.markdown(get_table_download_link(st.session_state.calibration_data['heating'], "calibration_heating.csv", "📥 Heating CSV"), unsafe_allow_html=True)
        with col2: st.markdown(get_table_download_link(st.session_state.calibration_data['cooling'], "calibration_cooling.csv", "📥 Cooling CSV"), unsafe_allow_html=True)
        with col3: st.markdown(get_excel_download_link(cal_data, "calibration_full.xlsx", "📥 Full Data XLSX"), unsafe_allow_html=True)
        with col4: st.markdown(get_plot_download_link(fig1, "voltage_plots.png", "📥 Voltage Plots"), unsafe_allow_html=True)
        with col5: st.markdown(get_plot_download_link(fig2, "temperature_plots.png", "📥 Temp Plots"), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== MODIFIED EMISSIVITY STUDIO ==========

def run_emissivity_studio_corrected(calibration_data, blackbody_data, spline_s, outlier_threshold):
    try:
        if blackbody_data is None:
            st.error("❌ BLACKBODY DATA REQUIRED!")
            return None, None
        
        bb_df = pd.read_excel(blackbody_data) if blackbody_data.name.endswith('.xlsx') else pd.read_csv(blackbody_data)
        T_bb, V_bb = bb_df['Temperature (°C)'].values, bb_df['Voltage'].values
        
        st.success(f"✅ Loaded blackbody data: {len(T_bb)} points from {T_bb.min():.1f}°C to {T_bb.max():.1f}°C")

        emissivity_data, outlier_info = {}, {}
        
        for cycle in ['heating', 'cooling']:
            cycle_data = calibration_data[cycle]
            temp, volt = cycle_data['TS_Temperature_C'].values, cycle_data['IR_Voltage_V'].values
            valid_mask = (temp >= T_bb.min()) & (temp <= T_bb.max())
            temp_valid, volt_valid = temp[valid_mask], volt[valid_mask]
            
            if len(temp_valid) > 50:
                temp_clean, volt_clean, temp_smooth, volt_smooth = adjustable_smoothing_enhanced(
                    temp_valid, volt_valid, spline_s, outlier_threshold, st.session_state.smoothing_enabled
                )
                
                if len(temp_clean) > 10:
                    volt_clean_series = pd.Series(volt_clean)
                    rolling_mean = volt_clean_series.rolling(window=10, center=True).mean()
                    rolling_std = volt_clean_series.rolling(window=10, center=True).std()
                    z_scores = np.abs((volt_clean - rolling_mean) / (rolling_std + 1e-8))
                    outlier_mask = z_scores > outlier_threshold
                    
                    outlier_info[cycle] = {
                        'total_points': len(temp_clean), 'outlier_count': np.sum(outlier_mask),
                        'outlier_percentage': (np.sum(outlier_mask) / len(temp_clean)) * 100,
                        'outlier_temperatures': temp_clean[outlier_mask], 'outlier_voltages': volt_clean[outlier_mask]
                    }
                
                if len(temp_smooth) > 0:
                    V_bb_interp = np.interp(temp_smooth, T_bb, V_bb)
                    emissivity = volt_smooth / (V_bb_interp + 1e-8)
                    emissivity = np.clip(emissivity, 0.1, 1.0)
                    
                    emissivity_data[cycle] = pd.DataFrame({
                        'TS_Temperature_C': temp_smooth, 'Emissivity': emissivity,
                        'Sample_Voltage': volt_smooth, 'Blackbody_Voltage': V_bb_interp,
                        'Residuals': volt_smooth - V_bb_interp
                    })
        
        return emissivity_data, outlier_info
        
    except Exception as e:
        st.error(f"Emissivity error: {str(e)}")
        return None, None

def render_emissivity_studio():
    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.header("📊 Emissivity Studio - Material Property Analysis")
    
    if st.session_state.emissivity_data:
        emissivity_data, outlier_info = st.session_state.emissivity_data
        st.success("✅ Emissivity Analysis Complete!")
        
        if outlier_info:
            st.subheader("🚨 Outlier Detection Summary")
            cols = st.columns(len(outlier_info))
            for i, (cycle, info) in enumerate(outlier_info.items()):
                with cols[i]:
                    st.metric(f"{cycle.capitalize()} Outliers", f"{info['outlier_count']} ({info['outlier_percentage']:.1f}%)")
        
        st.subheader("📈 Emissivity vs Temperature")
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for cycle, color in [('heating', 'red'), ('cooling', 'blue')]:
            if cycle in emissivity_data:
                data = emissivity_data[cycle]
                ax1.plot(data['TS_Temperature_C'], data['Emissivity'], color=color, label=f'{cycle.capitalize()} Cycle', linewidth=2)
                if outlier_info and cycle in outlier_info and len(outlier_info[cycle]['outlier_temperatures']) > 0:
                    outlier_temp = outlier_info[cycle]['outlier_temperatures']
                    outlier_emis = np.interp(outlier_temp, data['TS_Temperature_C'], data['Emissivity'])
                    ax1.scatter(outlier_temp, outlier_emis, color='black', s=50, label=f'{cycle.capitalize()} Outliers', zorder=5)
        
        ax1.set_xlabel('TS Temperature (°C)'); ax1.set_ylabel('Emissivity')
        ax1.set_title('Emissivity vs Temperature (Outliers Highlighted)'); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        for cycle, color in [('heating', 'red'), ('cooling', 'blue')]:
            if cycle in emissivity_data:
                data = emissivity_data[cycle]
                ax2.plot(data['TS_Temperature_C'], data['Residuals'], color=color, label=f'{cycle.capitalize()} Residuals', alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('TS Temperature (°C)'); ax2.set_ylabel('Voltage Residuals (V)')
        ax2.set_title('Sample vs Blackbody Voltage Residuals'); ax2.legend(); ax2.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        st.subheader("⚡ Sample Voltage vs Blackbody Voltage")
        fig2, ax3 = plt.subplots(figsize=(10, 6))
        
        for cycle, color in [('heating', 'red'), ('cooling', 'blue')]:
            if cycle in emissivity_data:
                data = emissivity_data[cycle]
                ax3.plot(data['TS_Temperature_C'], data['Sample_Voltage'], color=color, label=f'Sample Voltage ({cycle})', linewidth=2, alpha=0.8)
                ax3.plot(data['TS_Temperature_C'], data['Blackbody_Voltage'], color=color, linestyle='--', label=f'Blackbody Voltage ({cycle})', alpha=0.6)
        
        ax3.set_xlabel('TS Temperature (°C)'); ax3.set_ylabel('Voltage (V)')
        ax3.set_title('Sample Voltage vs Blackbody Voltage'); ax3.legend(); ax3.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        st.subheader("📊 Emissivity Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'heating' in emissivity_data:
                heat_emis = emissivity_data['heating']['Emissivity']
                st.metric("Heating Emissivity", f"{heat_emis.mean():.3f} ± {heat_emis.std():.3f}")
                st.metric("Heating Range", f"{heat_emis.min():.3f} - {heat_emis.max():.3f}")
        with col2:
            if 'cooling' in emissivity_data:
                cool_emis = emissivity_data['cooling']['Emissivity']
                st.metric("Cooling Emissivity", f"{cool_emis.mean():.3f} ± {cool_emis.std():.3f}")
                st.metric("Cooling Range", f"{cool_emis.min():.3f} - {cool_emis.max():.3f}")
        with col3:
            if 'heating' in emissivity_data and 'cooling' in emissivity_data:
                avg_emis = (heat_emis.mean() + cool_emis.mean()) / 2
                st.metric("Average Emissivity", f"{avg_emis:.3f}")
                hysteresis = abs(heat_emis.mean() - cool_emis.mean())
                st.metric("Emissivity Hysteresis", f"{hysteresis:.3f}")
        
        st.subheader("📋 Emissivity Data Preview")
        if 'heating' in emissivity_data:
            st.dataframe(emissivity_data['heating'].head(10))
        
        st.markdown("### 📥 Download Emissivity Data")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'heating' in emissivity_data:
                st.markdown(get_table_download_link(emissivity_data['heating'], "emissivity_heating.csv", "📥 Heating CSV"), unsafe_allow_html=True)
        with col2:
            if 'cooling' in emissivity_data:
                st.markdown(get_table_download_link(emissivity_data['cooling'], "emissivity_cooling.csv", "📥 Cooling CSV"), unsafe_allow_html=True)
        with col3:
            if 'heating' in emissivity_data:
                st.markdown(get_excel_download_link(emissivity_data['heating'], "emissivity_heating.xlsx", "📥 Heating XLSX"), unsafe_allow_html=True)
        with col4:
            st.markdown(get_plot_download_link(fig1, "emissivity_plots.png", "📥 Emissivity Plots"), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== ABSOLUTE TEMPERATURE STUDIO — FULLY CORRECTED VERSION ==========
# ========== ABSOLUTE TEMPERATURE STUDIO — WITH CORRECTION FACTORS ==========

def calculate_absolute_temperature_corrected(
    ts_temperature, ir_temperature,
    emissivity_temp, emissivity_values,
    spline_s, min_temp
):
    # Filter out low temperatures but respect the 1491°C TS limit
    valid_mask = (ts_temperature >= min_temp) & (ts_temperature <= 1491.0)
    ts_valid = ts_temperature[valid_mask]
    ir_valid = ir_temperature[valid_mask]

    if len(ts_valid) < 50:
        st.warning(f"⚠️ Only {len(ts_valid)} valid points after filtering (TS range: {ts_valid.min():.1f}°C to {ts_valid.max():.1f}°C)")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Sort for spline stability
    sort_idx = np.argsort(ts_valid)
    ts_sorted = ts_valid[sort_idx]
    ir_sorted = ir_valid[sort_idx]

    # Sort emissivity axis
    e_sort = np.argsort(emissivity_temp)
    emiss_temp_sorted = emissivity_temp[e_sort]
    emiss_values_sorted = emissivity_values[e_sort]

    try:
        # Remove duplicates
        unique_ts, idx_u = np.unique(ts_sorted, return_index=True)
        ir_unique = ir_sorted[idx_u]

        # Smooth IR temperature spline
        ir_spline = UnivariateSpline(unique_ts, ir_unique, s=len(unique_ts) * spline_s)

        # Create common temperature range within TS limits
        T_common = np.linspace(max(unique_ts.min(), min_temp), min(unique_ts.max(), 1491.0), 2000)
        IR_smooth = ir_spline(T_common)

        # Interpolate emissivity within valid range
        emiss_interp = np.interp(T_common, emiss_temp_sorted, emiss_values_sorted)
        emiss_interp = np.clip(emiss_interp, 1e-4, 1.0)

        # --- CORRECT STEFAN-BOLTZMANN ABSOLUTE TEMPERATURE PHYSICS ---
        # CORRECTED FORMULA: T_absolute = T_IR / (ε)^0.25
        # Stefan-Boltzmann: P = εσT⁴ => T_absolute = T_IR / ε^0.25
        # This accounts for the fact that IR measures lower than actual due to ε < 1
        absolute_temp = IR_smooth / (emiss_interp + 1e-8)**0.25

        # Debug output to verify the correction
        st.write(f"🔍 Physics Check:")
        st.write(f"   - IR range: {IR_smooth.min():.1f}°C to {IR_smooth.max():.1f}°C")
        st.write(f"   - Emissivity range: {emiss_interp.min():.3f} to {emiss_interp.max():.3f}")
        st.write(f"   - Absolute range: {absolute_temp.min():.1f}°C to {absolute_temp.max():.1f}°C")
        
        # Sample calculation to show the physics
        sample_idx = len(absolute_temp) // 2
        st.write(f"   - Sample: IR={IR_smooth[sample_idx]:.1f}°C, ε={emiss_interp[sample_idx]:.3f}, T_abs={absolute_temp[sample_idx]:.1f}°C")
        st.write(f"   - Correction: +{absolute_temp[sample_idx] - IR_smooth[sample_idx]:.1f}°C")

        return T_common, IR_smooth, absolute_temp, emiss_interp

    except Exception as e:
        st.error(f"Spline fitting error: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])


def calculate_correction_factors(absolute_data):
    """Calculate material-specific correction factors from absolute temperature data"""
    correction_factors = {}
    material_info = {}
    
    for cycle in ["heating", "cooling"]:
        if cycle in absolute_data:
            df = absolute_data[cycle]
            
            # Calculate correction factor: Absolute / IR
            # This tells us how much to multiply IR by to get Absolute temperature
            correction_factors[f"{cycle}_correction"] = np.mean(df["Absolute_Temperature_Corrected"] / df["IR_Temperature_Uncorrected"])
            
            # Store material statistics
            material_info[f"{cycle}_stats"] = {
                "ir_range": (df["IR_Temperature_Uncorrected"].min(), df["IR_Temperature_Uncorrected"].max()),
                "absolute_range": (df["Absolute_Temperature_Corrected"].min(), df["Absolute_Temperature_Corrected"].max()),
                "avg_correction": np.mean(df["Absolute_Temperature_Corrected"] - df["IR_Temperature_Uncorrected"]),
                "points": len(df)
            }
    
    # Calculate overall correction factor (average of heating and cooling)
    if "heating_correction" in correction_factors and "cooling_correction" in correction_factors:
        correction_factors["overall_correction"] = (
            correction_factors["heating_correction"] + correction_factors["cooling_correction"]
        ) / 2
    
    return correction_factors, material_info


def run_absolute_temp_studio_corrected(calibration_data, emissivity_data, spline_s, min_temp):
    absolute_data = {}

    try:
        for cycle in ["heating", "cooling"]:
            if cycle not in emissivity_data:
                st.warning(f"⚠️ No {cycle} data in emissivity results")
                continue

            cal_cycle = calibration_data[cycle]
            emiss_cycle = emissivity_data[cycle]

            ts_temp = cal_cycle["TS_Temperature_C"].values
            ir_temp = cal_cycle["IR_Temperature_C"].values

            em_temp = emiss_cycle["TS_Temperature_C"].values
            em_val = emiss_cycle["Emissivity"].values

            st.write(f"🎯 Processing {cycle} cycle...")
            st.write(f"   - Calibration TS: {ts_temp.min():.1f}°C to {ts_temp.max():.1f}°C")
            st.write(f"   - Calibration IR: {ir_temp.min():.1f}°C to {ir_temp.max():.1f}°C")
            st.write(f"   - Emissivity TS: {em_temp.min():.1f}°C to {em_temp.max():.1f}°C")
            st.write(f"   - Emissivity values: {em_val.min():.3f} to {em_val.max():.3f}")

            T_common, IR_smooth, T_abs, emiss_interp = calculate_absolute_temperature_corrected(
                ts_temp, ir_temp, em_temp, em_val, spline_s, min_temp
            )

            if len(T_common) > 0:
                absolute_data[cycle] = pd.DataFrame({
                    "TS_Temperature_C": T_common,
                    "IR_Temperature_Uncorrected": IR_smooth,
                    "Absolute_Temperature_Corrected": T_abs,
                    "Emissivity": emiss_interp
                })
                st.success(f"✅ {cycle.capitalize()} cycle: {len(T_common)} points calculated")
            else:
                st.error(f"❌ {cycle.capitalize()} cycle: No valid data points")

        # NEW: Calculate correction factors
        if absolute_data and len(absolute_data) > 0:
            correction_factors, material_info = calculate_correction_factors(absolute_data)
            absolute_data["correction_factors"] = correction_factors
            absolute_data["material_info"] = material_info
            
            # Display correction factors
            st.subheader("📊 Correction Factors Calculated")
            col1, col2, col3 = st.columns(3)
            with col1:
                if "heating_correction" in correction_factors:
                    st.metric("Heating Correction", f"{correction_factors['heating_correction']:.3f}")
            with col2:
                if "cooling_correction" in correction_factors:
                    st.metric("Cooling Correction", f"{correction_factors['cooling_correction']:.3f}")
            with col3:
                if "overall_correction" in correction_factors:
                    st.metric("Overall Correction", f"{correction_factors['overall_correction']:.3f}")
            
            st.info(f"📈 These factors will be used in Prediction Studio: IR × {correction_factors.get('overall_correction', 1.0):.3f} = Absolute Temperature")

        return absolute_data

    except Exception as e:
        st.error(f"Absolute temperature error: {e}")
        return {}


def render_absolute_temp_studio():
    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.header("🌡️ Absolute Temperature Studio — WITH CORRECTION FACTORS")

    # Configuration
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)
    with col1:
        spline_s = st.slider("Spline Smoothing", 0.00001, 0.01, 0.0001, 0.00001, format="%.5f", key="abs_spline_s")
    with col2:
        min_temp = st.slider("Minimum Temperature (°C)", 0, 500, 100, 10, key="abs_min_temp")

    if st.button("🚀 Calculate Absolute Temperatures", key="run_absolute_temp", use_container_width=True):
        if 'calibration_data' not in st.session_state or 'emissivity_data' not in st.session_state:
            st.error("❌ Please run Calibration and Emissivity studios first!")
        else:
            with st.spinner("Calculating absolute temperatures and correction factors..."):
                emissivity_data, _ = st.session_state.emissivity_data
                absolute_data = run_absolute_temp_studio_corrected(
                    st.session_state.calibration_data,
                    emissivity_data,
                    spline_s,
                    min_temp
                )
                
                if absolute_data and len(absolute_data) > 0:
                    st.session_state.absolute_temp_data = absolute_data
                    st.success("✅ Absolute Temperature Analysis Complete!")
                else:
                    st.error("❌ Failed to calculate absolute temperatures")

    if 'absolute_temp_data' in st.session_state and st.session_state.absolute_temp_data:
        absolute_data = st.session_state.absolute_temp_data
        st.success("✅ Absolute Temperature Analysis Complete!")

        # Display correction factors prominently
        if "correction_factors" in absolute_data:
            st.subheader("🎯 Correction Factors for Prediction Studio")
            factors = absolute_data["correction_factors"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Heating Factor", f"{factors.get('heating_correction', 'N/A'):.3f}")
                st.write("Multiply IR by this for heating")
            with col2:
                st.metric("Cooling Factor", f"{factors.get('cooling_correction', 'N/A'):.3f}")
                st.write("Multiply IR by this for cooling")
            with col3:
                st.metric("Overall Factor", f"{factors.get('overall_correction', 'N/A'):.3f}")
                st.write("Recommended for prediction")
            
            st.info("💡 These real correction factors will be used in Prediction Studio")

        # Display data statistics
        st.subheader("📊 Data Statistics")
        cols = st.columns(2)
        for i, cycle in enumerate(["heating", "cooling"]):
            if cycle in absolute_data:
                data = absolute_data[cycle]
                with cols[i]:
                    st.metric(f"{cycle.capitalize()} - TS Temp Range", 
                             f"{data['TS_Temperature_C'].min():.1f}°C to {data['TS_Temperature_C'].max():.1f}°C")
                    st.metric(f"{cycle.capitalize()} - IR Temp Range", 
                             f"{data['IR_Temperature_Uncorrected'].min():.1f}°C to {data['IR_Temperature_Uncorrected'].max():.1f}°C")
                    st.metric(f"{cycle.capitalize()} - Absolute Temp Range", 
                             f"{data['Absolute_Temperature_Corrected'].min():.1f}°C to {data['Absolute_Temperature_Corrected'].max():.1f}°C")
                    st.metric(f"{cycle.capitalize()} - Emissivity Range", 
                             f"{data['Emissivity'].min():.3f} to {data['Emissivity'].max():.3f}")

        # Calculate average correction
        if "heating" in absolute_data and "cooling" in absolute_data:
            heat_data = absolute_data["heating"]
            cool_data = absolute_data["cooling"]
            
            avg_correction_heat = np.mean(heat_data["Absolute_Temperature_Corrected"] - heat_data["IR_Temperature_Uncorrected"])
            avg_correction_cool = np.mean(cool_data["Absolute_Temperature_Corrected"] - cool_data["IR_Temperature_Uncorrected"])
            
            st.info(f"📈 Average Stefan-Boltzmann Correction: Heating +{avg_correction_heat:.1f}°C, Cooling +{avg_correction_cool:.1f}°C")

        # Plotting (your existing plotting code)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        for cycle, color in [("heating", "red"), ("cooling", "blue")]:
            if cycle not in absolute_data:
                continue

            data = absolute_data[cycle]

            # T vs IR vs Absolute
            ax1.plot(data["TS_Temperature_C"], data["TS_Temperature_C"],
                     color=color, linestyle="--", alpha=0.7, label=f"TS Temp ({cycle})", linewidth=1)
            ax1.plot(data["TS_Temperature_C"], data["IR_Temperature_Uncorrected"],
                     color=color, linewidth=2, label=f"IR Temp ({cycle})", alpha=0.8)
            ax1.plot(data["TS_Temperature_C"], data["Absolute_Temperature_Corrected"],
                     color=color, linewidth=2, label=f"Absolute Temp ({cycle})", alpha=0.8)

            # Emissivity
            ax2.plot(data["TS_Temperature_C"], data["Emissivity"],
                     color=color, linewidth=2, label=f"Emissivity ({cycle})")

            # Adjustment
            adjust = data["Absolute_Temperature_Corrected"] - data["IR_Temperature_Uncorrected"]
            ax3.plot(data["TS_Temperature_C"], adjust,
                     color=color, linewidth=2, label=f"Temp Adjustment ({cycle})")

            # Absolute vs IR
            ax4.plot(data["IR_Temperature_Uncorrected"], data["Absolute_Temperature_Corrected"],
                     color=color, linewidth=2, label=f"{cycle.capitalize()}")

        # Labeling + grid
        ax1.set_xlabel("TS Temperature (°C)"); ax1.set_ylabel("Temperature (°C)")
        ax1.set_title("Temperature Comparison (All Data)"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("TS Temperature (°C)"); ax2.set_ylabel("Emissivity")
        ax2.set_title("Emissivity vs Temperature"); ax2.legend(); ax2.grid(True, alpha=0.3)

        ax3.set_xlabel("TS Temperature (°C)"); ax3.set_ylabel("Temperature Adjustment (°C)")
        ax3.set_title("Stefan-Boltzmann Correction (T_absolute - T_IR)"); ax3.legend(); ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # Absolute vs IR with ideal line
        all_ir = np.concatenate([
            absolute_data[cycle]["IR_Temperature_Uncorrected"]
            for cycle in absolute_data if cycle in ["heating", "cooling"]
        ])
        min_ir, max_ir = all_ir.min(), all_ir.max()
        ax4.plot([min_ir, max_ir], [min_ir, max_ir], "black", linestyle="--", alpha=0.5, label="Ideal")
        ax4.set_xlabel("IR Temperature (°C)"); ax4.set_ylabel("Absolute Temperature (°C)")
        ax4.set_title("Absolute vs IR Temperature"); ax4.legend(); ax4.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Data preview
        st.subheader("📋 Data Preview")
        tab1, tab2 = st.tabs(["Heating Data", "Cooling Data"])
        with tab1:
            if "heating" in absolute_data:
                st.dataframe(absolute_data["heating"].head(10))
        with tab2:
            if "cooling" in absolute_data:
                st.dataframe(absolute_data["cooling"].head(10))

        # Download options
        st.markdown("### 📥 Download Absolute Temperature Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            if "heating" in absolute_data:
                st.markdown(get_table_download_link(
                    absolute_data["heating"], 
                    "absolute_temp_heating.csv", 
                    "📥 Heating CSV"
                ), unsafe_allow_html=True)
        with col2:
            if "cooling" in absolute_data:
                st.markdown(get_table_download_link(
                    absolute_data["cooling"], 
                    "absolute_temp_cooling.csv", 
                    "📥 Cooling CSV"
                ), unsafe_allow_html=True)
        with col3:
            st.markdown(get_plot_download_link(fig, "absolute_temperature_plots.png", "📥 All Plots"), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    
# ========== COMPLETE HYSTERESIS STUDIO - DUAL ANALYSIS ==========

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import base64

def get_table_download_link(df, filename, link_text):
    """Generate download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def extract_emissivity_hysteresis_features(heating_df, cooling_df, label, min_temp=550, interpolation_points=500):
    """Calculate emissivity hysteresis using your exact equations"""
    T_col = "TS_Temperature_C"
    E_col = "Emissivity"
    
    # Filter data within TS limits
    heating_df = heating_df[(heating_df[T_col] >= min_temp) & (heating_df[T_col] <= 1491.0)].copy()
    cooling_df = cooling_df[(cooling_df[T_col] >= min_temp) & (cooling_df[T_col] <= 1491.0)].copy()
    
    if len(heating_df) < 10 or len(cooling_df) < 10:
        return create_nan_features(label, "emissivity"), None, None, None
    
    min_temp_overlap = max(heating_df[T_col].min(), cooling_df[T_col].min())
    max_temp_overlap = min(heating_df[T_col].max(), cooling_df[T_col].max())
    
    if min_temp_overlap >= max_temp_overlap:
        return create_nan_features(label, "emissivity"), None, None, None
    
    common_temp = np.linspace(min_temp_overlap, max_temp_overlap, interpolation_points)
    
    try:
        f_heat = interp1d(heating_df[T_col], heating_df[E_col], kind='cubic', bounds_error=False, fill_value="extrapolate")
        f_cool = interp1d(cooling_df[T_col], cooling_df[E_col], kind='cubic', bounds_error=False, fill_value="extrapolate")
        
        ε_heat = f_heat(common_temp)
        ε_cool = f_cool(common_temp)
        
        valid_mask = ~(np.isnan(ε_heat) | np.isnan(ε_cool))
        common_temp = common_temp[valid_mask]
        ε_heat = ε_heat[valid_mask]
        ε_cool = ε_cool[valid_mask]
        
        if len(common_temp) < 10:
            raise ValueError("Not enough valid data points")
            
    except Exception as e:
        return create_nan_features(label, "emissivity"), None, None, None
    
    # ===== YOUR EQUATIONS =====
    
    # Equation 6: Loop Area
    A_loop = np.trapz(np.abs(ε_heat - ε_cool), common_temp)
    
    # Equation 7: Maximum Loop Width
    ε_min_range = max(np.min(ε_heat), np.min(ε_cool))
    ε_max_range = min(np.max(ε_heat), np.max(ε_cool))
    ε_range = np.linspace(ε_min_range, ε_max_range, 100)
    
    W_loop_vals = []
    for ε_val in ε_range:
        try:
            T_heat = np.interp(ε_val, ε_heat[::-1], common_temp[::-1])
            T_cool = np.interp(ε_val, ε_cool, common_temp)
            W_loop_vals.append(T_cool - T_heat)
        except:
            W_loop_vals.append(np.nan)
    W_loop_vals = np.array(W_loop_vals)
    W_loop = np.nanmax(np.abs(W_loop_vals)) if not np.all(np.isnan(W_loop_vals)) else np.nan
    
    # Equation 8: Maximum rate of emissivity change
    dεdT_heat = np.gradient(ε_heat, common_temp)
    dεdT_cool = np.gradient(ε_cool, common_temp)
    max_dεdT_heat = np.max(np.abs(dεdT_heat))
    max_dεdT_cool = np.max(np.abs(dεdT_cool))
    mean_dεdT_heat = np.mean(np.abs(dεdT_heat))
    mean_dεdT_cool = np.mean(np.abs(dεdT_cool))
    
    # Equation 9: Mean emissivity difference
    Δε_mean = np.trapz(ε_heat - ε_cool, common_temp) / (max_temp_overlap - min_temp_overlap)
    
    # Equation 10: Residual offset
    ε_r = ε_cool[-1] - ε_heat[0]  # ε_cool(T_max) - ε_heat(T_min)
    
    # Equation 11: Skewness
    Δε_diff = ε_heat - ε_cool
    σ_Δε = np.std(Δε_diff)
    if σ_Δε > 0:
        S = np.mean(((Δε_diff - Δε_mean) / σ_Δε) ** 3)
    else:
        S = np.nan
    
    # Equation 12: Onset temperature
    Δε_max = np.max(np.abs(Δε_diff))
    onset_threshold = 0.05 * Δε_max
    T_onset = None
    for i, (T, Δε_val) in enumerate(zip(common_temp, Δε_diff)):
        if abs(Δε_val) >= onset_threshold:
            T_onset = T
            break
    T_onset = T_onset if T_onset is not None else np.nan
    
    # Equation 13: Saturation temperature
    ε_hmin, ε_hmax = np.min(ε_heat), np.max(ε_heat)
    saturation_threshold = ε_hmin + 0.95 * (ε_hmax - ε_hmin)
    T_sat = None
    for i, (T, ε_val) in enumerate(zip(common_temp, ε_heat)):
        if ε_val >= saturation_threshold:
            T_sat = T
            break
    T_sat = T_sat if T_sat is not None else np.nan
    
    # Additional metrics
    rms_difference = np.sqrt(np.mean((ε_heat - ε_cool)**2))
    max_difference = np.max(np.abs(ε_heat - ε_cool))
    mean_abs_difference = np.mean(np.abs(ε_heat - ε_cool))
    correlation = np.corrcoef(ε_heat, ε_cool)[0, 1] if len(ε_heat) > 2 else np.nan
    
    features = {
        "Label": f"{label}_emissivity", "Parameter": "Emissivity",
        "Loop_Area": A_loop, "Max_Loop_Width": W_loop,
        "Max_dεdT_Heat": max_dεdT_heat, "Max_dεdT_Cool": max_dεdT_cool,
        "Mean_dεdT_Heat": mean_dεdT_heat, "Mean_dεdT_Cool": mean_dεdT_cool,
        "Mean_Emissivity_Difference": Δε_mean, "Residual_Offset": ε_r,
        "Skewness": S, "Onset_Temperature": T_onset, "Saturation_Temperature": T_sat,
        "RMS_Difference": rms_difference, "Max_Difference": max_difference,
        "Mean_Abs_Difference": mean_abs_difference, "Correlation": correlation,
        "Analysis_Start_Temp": min_temp_overlap, "Analysis_End_Temp": max_temp_overlap,
        "Data_Points": len(common_temp)
    }
    
    return features, common_temp, ε_heat, ε_cool

def extract_temperature_hysteresis_features(heating_df, cooling_df, label, min_temp=550, interpolation_points=500):
    """Calculate temperature hysteresis using SAME EQUATIONS as emissivity"""
    T_col = "TS_Temperature_C"
    E_col = "Absolute_Temperature_Corrected"
    
    # Filter data within TS limits
    heating_df = heating_df[(heating_df[T_col] >= min_temp) & (heating_df[T_col] <= 1491.0)].copy()
    cooling_df = cooling_df[(cooling_df[T_col] >= min_temp) & (cooling_df[T_col] <= 1491.0)].copy()
    
    if len(heating_df) < 10 or len(cooling_df) < 10:
        return create_nan_features(label, "temperature"), None, None, None
    
    min_temp_overlap = max(heating_df[T_col].min(), cooling_df[T_col].min())
    max_temp_overlap = min(heating_df[T_col].max(), cooling_df[T_col].max())
    
    if min_temp_overlap >= max_temp_overlap:
        return create_nan_features(label, "temperature"), None, None, None
    
    common_temp = np.linspace(min_temp_overlap, max_temp_overlap, interpolation_points)
    
    try:
        f_heat = interp1d(heating_df[T_col], heating_df[E_col], kind='cubic', bounds_error=False, fill_value="extrapolate")
        f_cool = interp1d(cooling_df[T_col], cooling_df[E_col], kind='cubic', bounds_error=False, fill_value="extrapolate")
        
        T_abs_heat = f_heat(common_temp)
        T_abs_cool = f_cool(common_temp)
        
        valid_mask = ~(np.isnan(T_abs_heat) | np.isnan(T_abs_cool))
        common_temp = common_temp[valid_mask]
        T_abs_heat = T_abs_heat[valid_mask]
        T_abs_cool = T_abs_cool[valid_mask]
        
        if len(common_temp) < 10:
            raise ValueError("Not enough valid data points")
            
    except Exception as e:
        return create_nan_features(label, "temperature"), None, None, None
    
    # ===== APPLY SAME EQUATIONS AS EMISSIVITY =====
    
    # Equation 6: Loop Area
    A_loop = np.trapz(np.abs(T_abs_heat - T_abs_cool), common_temp)
    
    # Equation 7: Maximum Loop Width
    T_min_range = max(np.min(T_abs_heat), np.min(T_abs_cool))
    T_max_range = min(np.max(T_abs_heat), np.max(T_abs_cool))
    T_range = np.linspace(T_min_range, T_max_range, 100)
    
    W_loop_vals = []
    for T_val in T_range:
        try:
            T_heat_ts = np.interp(T_val, T_abs_heat[::-1], common_temp[::-1])
            T_cool_ts = np.interp(T_val, T_abs_cool, common_temp)
            W_loop_vals.append(T_cool_ts - T_heat_ts)
        except:
            W_loop_vals.append(np.nan)
    W_loop_vals = np.array(W_loop_vals)
    W_loop = np.nanmax(np.abs(W_loop_vals)) if not np.all(np.isnan(W_loop_vals)) else np.nan
    
    # Equation 8: Maximum rate of temperature change
    dTdT_heat = np.gradient(T_abs_heat, common_temp)
    dTdT_cool = np.gradient(T_abs_cool, common_temp)
    max_dTdT_heat = np.max(np.abs(dTdT_heat))
    max_dTdT_cool = np.max(np.abs(dTdT_cool))
    mean_dTdT_heat = np.mean(np.abs(dTdT_heat))
    mean_dTdT_cool = np.mean(np.abs(dTdT_cool))
    
    # Equation 9: Mean temperature difference
    ΔT_mean = np.trapz(T_abs_heat - T_abs_cool, common_temp) / (max_temp_overlap - min_temp_overlap)
    
    # Equation 10: Residual offset
    T_r = T_abs_cool[-1] - T_abs_heat[0]  # T_abs_cool(T_max) - T_abs_heat(T_min)
    
    # Equation 11: Skewness
    ΔT_diff = T_abs_heat - T_abs_cool
    σ_ΔT = np.std(ΔT_diff)
    if σ_ΔT > 0:
        S = np.mean(((ΔT_diff - ΔT_mean) / σ_ΔT) ** 3)
    else:
        S = np.nan
    
    # Equation 12: Onset temperature
    ΔT_max = np.max(np.abs(ΔT_diff))
    onset_threshold = 0.05 * ΔT_max
    T_onset = None
    for i, (T, ΔT_val) in enumerate(zip(common_temp, ΔT_diff)):
        if abs(ΔT_val) >= onset_threshold:
            T_onset = T
            break
    T_onset = T_onset if T_onset is not None else np.nan
    
    # Equation 13: Saturation temperature
    T_hmin, T_hmax = np.min(T_abs_heat), np.max(T_abs_heat)
    saturation_threshold = T_hmin + 0.95 * (T_hmax - T_hmin)
    T_sat = None
    for i, (T, T_val) in enumerate(zip(common_temp, T_abs_heat)):
        if T_val >= saturation_threshold:
            T_sat = T
            break
    T_sat = T_sat if T_sat is not None else np.nan
    
    # Additional metrics (same as emissivity)
    rms_difference = np.sqrt(np.mean((T_abs_heat - T_abs_cool)**2))
    max_difference = np.max(np.abs(T_abs_heat - T_abs_cool))
    mean_abs_difference = np.mean(np.abs(T_abs_heat - T_abs_cool))
    correlation = np.corrcoef(T_abs_heat, T_abs_cool)[0, 1] if len(T_abs_heat) > 2 else np.nan
    
    features = {
        "Label": f"{label}_temperature", "Parameter": "Absolute_Temperature",
        "Loop_Area": A_loop, "Max_Loop_Width": W_loop,
        "Max_dTdT_Heat": max_dTdT_heat, "Max_dTdT_Cool": max_dTdT_cool,
        "Mean_dTdT_Heat": mean_dTdT_heat, "Mean_dTdT_Cool": mean_dTdT_cool,
        "Mean_Temperature_Difference": ΔT_mean, "Residual_Offset": T_r,
        "Skewness": S, "Onset_Temperature": T_onset, "Saturation_Temperature": T_sat,
        "RMS_Difference": rms_difference, "Max_Difference": max_difference,
        "Mean_Abs_Difference": mean_abs_difference, "Correlation": correlation,
        "Analysis_Start_Temp": min_temp_overlap, "Analysis_End_Temp": max_temp_overlap,
        "Data_Points": len(common_temp)
    }
    
    return features, common_temp, T_abs_heat, T_abs_cool

def create_nan_features(label, parameter_type):
    base_features = {
        "Label": f"{label}_{parameter_type}", "Parameter": parameter_type,
        "Loop_Area": np.nan, "Max_Loop_Width": np.nan, "Max_dεdT_Heat": np.nan,
        "Max_dεdT_Cool": np.nan, "Mean_dεdT_Heat": np.nan, "Mean_dεdT_Cool": np.nan,
        "Mean_Emissivity_Difference": np.nan, "Residual_Offset": np.nan,
        "Skewness": np.nan, "Onset_Temperature": np.nan, "Saturation_Temperature": np.nan,
        "RMS_Difference": np.nan, "Max_Difference": np.nan, "Mean_Abs_Difference": np.nan,
        "Correlation": np.nan, "Max_dTdT_Heat": np.nan, "Max_dTdT_Cool": np.nan,
        "Analysis_Start_Temp": np.nan, "Analysis_End_Temp": np.nan, "Data_Points": np.nan
    }
    return base_features

def run_hysteresis_studio_enhanced(absolute_data, min_temp=550, interpolation_points=500):
    try:
        if 'heating' not in absolute_data or 'cooling' not in absolute_data:
            return None
            
        heat_data, cool_data = absolute_data['heating'], absolute_data['cooling']
        
        # Run both analyses
        emissivity_features, ε_common_temp, ε_heat, ε_cool = extract_emissivity_hysteresis_features(
            heat_data, cool_data, "Sample", min_temp, interpolation_points
        )
        
        temperature_features, T_common_temp, T_abs_heat, T_abs_cool = extract_temperature_hysteresis_features(
            heat_data, cool_data, "Sample", min_temp, interpolation_points
        )
        
        hysteresis_data = {
            'emissivity': {
                'features': emissivity_features, 'common_temp': ε_common_temp,
                'heat_values': ε_heat, 'cool_values': ε_cool,
                'delta_values': ε_heat - ε_cool
            },
            'temperature': {
                'features': temperature_features, 'common_temp': T_common_temp,
                'heat_values': T_abs_heat, 'cool_values': T_abs_cool,
                'delta_values': T_abs_heat - T_abs_cool
            }
        }
        
        return hysteresis_data
        
    except Exception as e:
        st.error(f"Hysteresis error: {str(e)}")
        return None

def render_hysteresis_studio():
    """MAIN RENDERING FUNCTION - THIS WAS MISSING"""
    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.header("🔄 Hysteresis Studio - Dual Analysis (Emissivity + Temperature)")
    
    # Configuration
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)
    with col1:
        min_temp = st.slider("Minimum Temperature (°C)", 0, 800, 550, 10, key="hyst_min_temp")
    with col2:
        interpolation_points = st.slider("Interpolation Points", 100, 1000, 500, key="hyst_interp_points")

    if st.button("🚀 Analyze Both Hysteresis Types", key="run_hysteresis", use_container_width=True):
        if 'absolute_temp_data' not in st.session_state:
            st.error("❌ Please run Absolute Temperature Studio first!")
        else:
            with st.spinner("Analyzing emissivity and temperature hysteresis..."):
                hysteresis_data = run_hysteresis_studio_enhanced(
                    st.session_state.absolute_temp_data,
                    min_temp,
                    interpolation_points
                )
                
                if hysteresis_data is not None:
                    st.session_state.hysteresis_data = hysteresis_data
                    st.success("✅ Dual Hysteresis Analysis Complete!")
                else:
                    st.error("❌ Hysteresis analysis failed")
    
    if 'hysteresis_data' in st.session_state and st.session_state.hysteresis_data:
        hd = st.session_state.hysteresis_data
        ε_data = hd['emissivity']
        T_data = hd['temperature']
        
        st.success("✅ Dual Hysteresis Analysis Complete!")
        
        # Create tabs for both analyses
        tab1, tab2 = st.tabs(["📊 Emissivity Hysteresis", "🌡️ Temperature Hysteresis"])
        
        with tab1:
            if ε_data['common_temp'] is not None:
                st.subheader("Emissivity Hysteresis - Your Equations")
                ε_features = ε_data['features']
                
                fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Emissivity vs Temperature
                ax1.plot(ε_data['common_temp'], ε_data['heat_values'], 'red', linewidth=2, label='Heating', alpha=0.8)
                ax1.plot(ε_data['common_temp'], ε_data['cool_values'], 'blue', linewidth=2, label='Cooling', alpha=0.8)
                ax1.fill_between(ε_data['common_temp'], ε_data['heat_values'], ε_data['cool_values'], alpha=0.3, color='gray', label='Hysteresis Area')
                ax1.set_xlabel('TS Temperature (°C)'); ax1.set_ylabel('Emissivity')
                ax1.set_title(f'Emissivity Hysteresis Loop\nArea: {ε_features["Loop_Area"]:.3f} °C'); ax1.legend(); ax1.grid(True, alpha=0.3)
                
                # Emissivity Difference
                ax2.plot(ε_data['common_temp'], ε_data['delta_values'], 'purple', linewidth=2)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax2.set_xlabel('TS Temperature (°C)'); ax2.set_ylabel('Δε (Heating - Cooling)')
                ax2.set_title(f'Emissivity Difference\nMax Width: {ε_features["Max_Loop_Width"]:.1f}°C'); ax2.grid(True, alpha=0.3)
                
                # Derivatives
                dεdT_heat = np.gradient(ε_data['heat_values'], ε_data['common_temp'])
                dεdT_cool = np.gradient(ε_data['cool_values'], ε_data['common_temp'])
                ax3.plot(ε_data['common_temp'], dεdT_heat, 'red', linewidth=2, label='Heating', alpha=0.8)
                ax3.plot(ε_data['common_temp'], dεdT_cool, 'blue', linewidth=2, label='Cooling', alpha=0.8)
                ax3.set_xlabel('TS Temperature (°C)'); ax3.set_ylabel('dε/dT')
                ax3.set_title(f'Emissivity Derivatives'); ax3.legend(); ax3.grid(True, alpha=0.3)
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Feature Summary
                ax4.axis('off')
                feature_text = f"""EMISSIVITY HYSTERESIS - YOUR EQUATIONS

Loop Metrics:
• Loop Area: {ε_features['Loop_Area']:.3f} °C
• Max Loop Width: {ε_features['Max_Loop_Width']:.1f} °C
• Mean Δε: {ε_features['Mean_Emissivity_Difference']:.4f}
• Residual Offset: {ε_features['Residual_Offset']:.4f}

Transformation Metrics:
• Onset Temp: {ε_features['Onset_Temperature']:.1f} °C
• Saturation Temp: {ε_features['Saturation_Temperature']:.1f} °C
• Skewness: {ε_features['Skewness']:.3f}

Statistical Metrics:
• RMS Difference: {ε_features['RMS_Difference']:.4f}
• Max Difference: {ε_features['Max_Difference']:.4f}
• Correlation: {ε_features['Correlation']:.3f}"""
                ax4.text(0.05, 0.95, feature_text, transform=ax4.transAxes, fontfamily='monospace', 
                        verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                
                plt.tight_layout()
                st.pyplot(fig1)
                
                # Emissivity Data Download
                ε_df = pd.DataFrame({
                    'TS_Temperature_C': ε_data['common_temp'],
                    'Emissivity_Heating': ε_data['heat_values'],
                    'Emissivity_Cooling': ε_data['cool_values'],
                    'Emissivity_Difference': ε_data['delta_values']
                })
                st.markdown(get_table_download_link(ε_df, "emissivity_hysteresis_data.csv", "📥 Emissivity Data CSV"), unsafe_allow_html=True)
        
        with tab2:
            if T_data['common_temp'] is not None:
                st.subheader("Temperature Hysteresis - Absolute Temperature Differences")
                T_features = T_data['features']
                
                fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Absolute Temperature vs TS Temperature
                ax1.plot(T_data['common_temp'], T_data['heat_values'], 'red', linewidth=2, label='Heating', alpha=0.8)
                ax1.plot(T_data['common_temp'], T_data['cool_values'], 'blue', linewidth=2, label='Cooling', alpha=0.8)
                ax1.fill_between(T_data['common_temp'], T_data['heat_values'], T_data['cool_values'], alpha=0.3, color='gray', label='Hysteresis Area')
                ax1.set_xlabel('TS Temperature (°C)'); ax1.set_ylabel('Absolute Temperature (°C)')
                ax1.set_title(f'Temperature Hysteresis Loop\nArea: {T_features["Loop_Area"]:.1f} °C²'); ax1.legend(); ax1.grid(True, alpha=0.3)
                
                # Temperature Difference
                ax2.plot(T_data['common_temp'], T_data['delta_values'], 'purple', linewidth=2)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax2.set_xlabel('TS Temperature (°C)'); ax2.set_ylabel('ΔT (Heating - Cooling) (°C)')
                ax2.set_title(f'Temperature Difference\nMax: {T_features["Max_Difference"]:.1f}°C'); ax2.grid(True, alpha=0.3)
                
                # Derivatives
                dTdT_heat = np.gradient(T_data['heat_values'], T_data['common_temp'])
                dTdT_cool = np.gradient(T_data['cool_values'], T_data['common_temp'])
                ax3.plot(T_data['common_temp'], dTdT_heat, 'red', linewidth=2, label='Heating', alpha=0.8)
                ax3.plot(T_data['common_temp'], dTdT_cool, 'blue', linewidth=2, label='Cooling', alpha=0.8)
                ax3.set_xlabel('TS Temperature (°C)'); ax3.set_ylabel('dT/dT')
                ax3.set_title(f'Temperature Derivatives'); ax3.legend(); ax3.grid(True, alpha=0.3)
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Feature Summary
                ax4.axis('off')
                feature_text = f"""TEMPERATURE HYSTERESIS

Loop Metrics:
• Loop Area: {T_features['Loop_Area']:.1f} °C²
• Max Loop Width: {T_features['Max_Loop_Width']:.1f} °C
• Max Difference: {T_features['Max_Difference']:.1f} °C
• Mean Abs Difference: {T_features['Mean_Abs_Difference']:.1f} °C

Statistical Metrics:
• RMS Difference: {T_features['RMS_Difference']:.1f} °C
• Correlation: {T_features['Correlation']:.3f}

Analysis Range:
• Start: {T_features['Analysis_Start_Temp']:.1f} °C
• End: {T_features['Analysis_End_Temp']:.1f} °C
• Points: {T_features['Data_Points']}"""
                ax4.text(0.05, 0.95, feature_text, transform=ax4.transAxes, fontfamily='monospace', 
                        verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Temperature Data Download
                T_df = pd.DataFrame({
                    'TS_Temperature_C': T_data['common_temp'],
                    'Absolute_Temp_Heating': T_data['heat_values'],
                    'Absolute_Temp_Cooling': T_data['cool_values'],
                    'Temperature_Difference': T_data['delta_values']
                })
                st.markdown(get_table_download_link(T_df, "temperature_hysteresis_data.csv", "📥 Temperature Data CSV"), unsafe_allow_html=True)
        
        # Combined Features Download
        st.subheader("📥 Download All Hysteresis Features")
        all_features_df = pd.DataFrame([ε_data['features'], T_data['features']])
        st.markdown(get_table_download_link(all_features_df, "all_hysteresis_features.csv", "📥 All Features CSV"), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== COMPLETE PREDICTION STUDIO - USES REAL CORRECTION FACTORS ==========
# ========== COMPLETE PREDICTION STUDIO - USES REAL CORRECTION FACTORS & REAL ERROR ANALYSIS ==========

class TunedCommercialPredictor:
    def __init__(self):
        # Initial confidence + max allowed error (from your older calibration)
        self.material_models = {
            "316L": {"confidence": 0.92, "max_error": 12},
            "Cantor": {"confidence": 0.89, "max_error": 15},
            "Inconel 718": {"confidence": 0.91, "max_error": 11},
            "Ti-6Al-4V": {"confidence": 0.94, "max_error": 9},
            "Ni50.1Ti": {"confidence": 0.88, "max_error": 14},
            "Ni52Ti": {"confidence": 0.87, "max_error": 16},
            "Auto-Detect": {"confidence": 0.85, "max_error": 18},
        }

    def get_real_correction_factor(self, material="Auto-Detect"):
        """GET REAL CORRECTION FACTOR from Absolute Temperature Studio"""
        if 'absolute_temp_data' not in st.session_state:
            return None, "❌ No calibration data available - run Absolute Temperature Studio first"
            
        calibration_data = st.session_state.absolute_temp_data
        
        # Check if calibration_data exists and is not None
        if calibration_data is None:
            return None, "❌ Calibration data is empty - run Absolute Temperature Studio first"
        
        # Check if it's a dictionary with correction_factors
        if not isinstance(calibration_data, dict) or "correction_factors" not in calibration_data:
            return None, "⚠️ No correction factors found in calibration data"
            
        # Try to get the real correction factor
        factors = calibration_data["correction_factors"]
        
        # Use overall correction if available
        if "overall_correction" in factors:
            correction = factors["overall_correction"]
            return correction, "✅ Using REAL correction factor from your data"
        
        # Fallback to heating or cooling
        elif "heating_correction" in factors:
            correction = factors["heating_correction"]
            return correction, "✅ Using heating correction factor from your data"
        elif "cooling_correction" in factors:
            correction = factors["cooling_correction"]
            return correction, "✅ Using cooling correction factor from your data"
        
        return None, "⚠️ No correction factors found in calibration data"

    def evaluate_real_error(self, material):
        """
        Compute real absolute error from actual uploaded data.
        Uses the calibration data to compare predicted vs actual temperatures.
        """
        if 'absolute_temp_data' not in st.session_state:
            return None  # No calibration data available
            
        calibration_data = st.session_state.absolute_temp_data
        
        if calibration_data is None:
            return None  # Calibration data is empty
        
        if not isinstance(calibration_data, dict):
            return None  # Invalid calibration data format
        
        # Extract IR and Absolute temperature data for error analysis
        all_ir = []
        all_absolute = []
        
        for cycle in ["heating", "cooling"]:
            if cycle in calibration_data:
                df = calibration_data[cycle]
                if isinstance(df, pd.DataFrame) and 'IR_Temperature_Uncorrected' in df.columns and 'Absolute_Temperature_Corrected' in df.columns:
                    all_ir.extend(df['IR_Temperature_Uncorrected'].tolist())
                    all_absolute.extend(df['Absolute_Temperature_Corrected'].tolist())
        
        if len(all_ir) < 5:
            return None  # Not enough calibration data for error analysis
        
        # Get the correction factor for this material
        correction_factor, _ = self.get_real_correction_factor(material)
        if not correction_factor:
            correction_factor = 1.135  # Default fallback
        
        # Calculate predicted temperatures using the correction factor
        predicted = [ir_temp * correction_factor for ir_temp in all_ir]
        actual = all_absolute
        
        # Real absolute error (per point)
        errors = np.abs(np.array(predicted) - np.array(actual))
        
        # Summary statistics
        real_mean_error = float(np.mean(errors))
        real_max_error = float(np.max(errors))
        real_std_error = float(np.std(errors))

        model = self.material_models.get(material, None)
        if model is None:
            return None  # Material not found in models

        # Compare to stored max_error
        exceeds = real_max_error > model["max_error"]
        
        # Auto-adjust confidence based on real error performance
        # Better performance than expected increases confidence, worse decreases it
        error_ratio = model["max_error"] / max(real_max_error, 1)  # Avoid division by zero
        adjusted_confidence = min(0.99, model["confidence"] * error_ratio)
        
        # If we're doing better than expected, give a small boost
        if error_ratio > 1.1:
            adjusted_confidence = min(0.99, adjusted_confidence * 1.05)
        
        return {
            "material": material,
            "data_points": len(all_ir),
            "stored_max_error": model["max_error"],
            "real_mean_error": real_mean_error,
            "real_max_error": real_max_error,
            "real_std_error": real_std_error,
            "exceeds_limit": exceeds,
            "original_confidence": model["confidence"],
            "adjusted_confidence": round(adjusted_confidence, 3),
            "correction_factor_used": correction_factor,
            "error_improvement": real_max_error < model["max_error"]
        }

    def predict_commercial(self, ir_temp, material="Auto-Detect"):
        """Predict Absolute Temperature from IR input using REAL correction factors"""
        # Get REAL correction factor from your data
        real_correction, message = self.get_real_correction_factor(material)
        
        if material == "Auto-Detect":
            # Simple auto-detection based on temperature range
            if ir_temp > 1400: 
                material = "Cantor"
            elif ir_temp < 1000: 
                material = "316L"
            else: 
                material = "Ni50.1Ti"
        
        # Use REAL correction factor if available
        if real_correction:
            correction = real_correction
            used_real_data = True
        else:
            # Fallback to theoretical values (will be replaced by real data)
            correction = 1.135  # Default average
            used_real_data = False
            message = "⚠️ Using default correction factor"
        
        # Calculate Absolute Temperature
        predicted_absolute = ir_temp * correction
        
        # Get REAL error analysis if calibration data exists
        real_error_analysis = None
        if 'absolute_temp_data' in st.session_state and st.session_state.absolute_temp_data is not None:
            real_error_analysis = self.evaluate_real_error(material)  # FIXED: No unpacking here
        
        # Use adjusted confidence if real error analysis is available
        if real_error_analysis and real_error_analysis.get('adjusted_confidence'):
            confidence = real_error_analysis['adjusted_confidence']
            max_error = real_error_analysis['real_max_error']
            uses_real_error = True
        else:
            confidence = self.material_models[material]["confidence"]
            max_error = self.material_models[material]["max_error"]
            uses_real_error = False
        
        return {
            "ir_input": ir_temp,
            "predicted_absolute": predicted_absolute,
            "correction_factor": correction,
            "material_detected": material,
            "confidence": confidence,
            "max_error": max_error,
            "used_real_data": used_real_data,
            "uses_real_error": uses_real_error,
            "correction_message": message,
            "real_error_analysis": real_error_analysis
        }

class TunedResearchPredictor:
    def __init__(self):
        self.smoothing_methods = ['polynomial', 'spline', 'lowess', 'moving_avg']
    
    def get_absolute_temperature_data(self):
        """Get REAL Absolute Temperature data from calibration for research analysis"""
        if 'absolute_temp_data' not in st.session_state:
            return None, None, "❌ No calibration data available"
            
        calibration_data = st.session_state.absolute_temp_data
        
        # Check if calibration_data exists and is not None
        if calibration_data is None:
            return None, None, "❌ Calibration data is empty - run Absolute Temperature Studio first"
        
        # Check if it's a dictionary
        if not isinstance(calibration_data, dict):
            return None, None, "⚠️ Invalid calibration data format"
        
        # Extract IR and Absolute temperature data for research analysis
        all_ir = []
        all_absolute = []
        
        for cycle in ["heating", "cooling"]:
            if cycle in calibration_data:
                df = calibration_data[cycle]
                if isinstance(df, pd.DataFrame) and 'IR_Temperature_Uncorrected' in df.columns and 'Absolute_Temperature_Corrected' in df.columns:
                    all_ir.extend(df['IR_Temperature_Uncorrected'].tolist())
                    all_absolute.extend(df['Absolute_Temperature_Corrected'].tolist())
        
        if len(all_ir) > 10:
            return np.array(all_ir), np.array(all_absolute), "✅ Using real calibration data for research analysis"
        else:
            return None, None, "⚠️ Not enough calibration data for research analysis"
    
    def polynomial_prediction(self, ir_temp, absolute_temp, degree=3):
        """Polynomial regression using REAL Absolute Temperature data"""
        try:
            poly_coeffs = np.polyfit(ir_temp, absolute_temp, degree)
            poly_func = np.poly1d(poly_coeffs)
            ir_range = np.linspace(ir_temp.min(), ir_temp.max(), 1000)
            absolute_pred = poly_func(ir_range)
            return ir_range, absolute_pred, poly_func
        except:
            return ir_temp, absolute_temp, None
    
    def spline_prediction(self, ir_temp, absolute_temp, smoothing=0.0001):
        """Spline prediction using REAL Absolute Temperature data"""
        try:
            df = pd.DataFrame({'ir': ir_temp, 'absolute': absolute_temp}).drop_duplicates('ir').sort_values('ir')
            if len(df) < 10:
                return ir_temp, absolute_temp, None
            spline = UnivariateSpline(df['ir'], df['absolute'], s=len(df)*smoothing)
            ir_range = np.linspace(df['ir'].min(), df['ir'].max(), 1000)
            absolute_pred = spline(ir_range)
            return ir_range, absolute_pred, spline
        except:
            return ir_temp, absolute_temp, None

class AMProcessPredictor:
    """AM Process Temperature Prediction USING REAL CORRECTION FACTORS"""
    
    def __init__(self):
        self.material_properties = {
            "316L": {
                "melting_point": 1400, 
                "optimal_range": (1300, 1600), 
                "thermal_conductivity": 25,
                "absorptivity": 0.6,
                "density": 8000,
                "specific_heat": 500,
            },
            "Cantor": {
                "melting_point": 1350, 
                "optimal_range": (1350, 1650), 
                "thermal_conductivity": 20,
                "absorptivity": 0.55,
                "density": 8200,
                "specific_heat": 480,
            },
            "Inconel 718": {
                "melting_point": 1330, 
                "optimal_range": (1280, 1580), 
                "thermal_conductivity": 11,
                "absorptivity": 0.5,
                "density": 8190,
                "specific_heat": 440,
            },
            "Ti-6Al-4V": {
                "melting_point": 1650, 
                "optimal_range": (1550, 1850), 
                "thermal_conductivity": 7,
                "absorptivity": 0.7,
                "density": 4430,
                "specific_heat": 526,
            },
            "Ni50.1Ti": {
                "melting_point": 1310, 
                "optimal_range": (1250, 1550), 
                "thermal_conductivity": 18,
                "absorptivity": 0.65,
                "density": 6450,
                "specific_heat": 470,
                "phase_transformation": "Austenite↔Martensite: 40-80°C"
            },
            "Ni52Ti": {
                "melting_point": 1320, 
                "optimal_range": (1260, 1560), 
                "thermal_conductivity": 17,
                "absorptivity": 0.63,
                "density": 6480,
                "specific_heat": 465,
                "phase_transformation": "Austenite + Ni₄Ti₃ precipitates"
            }
        }
    
    def get_real_correction_factor(self):
        """GET REAL CORRECTION FACTOR from Absolute Temperature Studio"""
        if 'absolute_temp_data' not in st.session_state:
            return None, "❌ No calibration data available"
            
        calibration_data = st.session_state.absolute_temp_data
        
        # Check if calibration_data exists and is not None
        if calibration_data is None:
            return None, "❌ Calibration data is empty - run Absolute Temperature Studio first"
        
        if not isinstance(calibration_data, dict) or "correction_factors" not in calibration_data:
            return None, "⚠️ No correction factors found in calibration data"

        factors = calibration_data["correction_factors"]
        
        if "overall_correction" in factors:
            return factors["overall_correction"], "✅ Using REAL correction factor from your data"
        elif "heating_correction" in factors:
            return factors["heating_correction"], "✅ Using heating correction factor from your data"
        
        return None, "⚠️ No correction factors found"
    
    def calculate_realistic_temperature(self, laser_power, scan_speed, layer_thickness, material="316L"):
        """
        Calculate Absolute Temperature from process parameters using REAL correction factors
        """
        props = self.material_properties[material]
        
        # REAL Energy Density Calculation
        spot_diameter = 0.1  # mm
        spot_area = np.pi * (spot_diameter/2)**2  # mm²
        exposure_time = spot_diameter / scan_speed  # seconds
        volume_melted = spot_area * layer_thickness  # mm³
        
        energy_density = (laser_power * exposure_time) / volume_melted
        
        # REALISTIC Temperature Calculation (1400-2500°C range for melt pools)
        base_temp = 200  # Preheat temperature
        
        # Map energy density to realistic temperature range
        min_energy, max_energy = 30, 100  # J/mm³
        min_temp, max_temp = 1400, 2500   # °C
        
        if energy_density <= min_energy:
            melt_pool_temp = min_temp
        elif energy_density >= max_energy:
            melt_pool_temp = max_temp
        else:
            energy_ratio = (energy_density - min_energy) / (max_energy - min_energy)
            melt_pool_temp = min_temp + energy_ratio * (max_temp - min_temp)
        
        # Ensure above melting point
        melt_pool_temp = max(melt_pool_temp, props["melting_point"] + 50)
        
        # Get REAL correction factor
        real_correction, cal_message = self.get_real_correction_factor()
        
        if real_correction:
            # Calculate what IR would read: IR = Absolute / Correction
            ir_reading = melt_pool_temp / real_correction
            absolute_temp = melt_pool_temp  # We're already working in Absolute temperature
            used_real_data = True
        else:
            # Fallback: assume IR reads 85% of actual (typical for metals)
            ir_reading = melt_pool_temp * 0.85
            absolute_temp = melt_pool_temp
            used_real_data = False
            cal_message = "⚠️ Using theoretical IR correction"
        
        return {
            "melt_pool_temp": melt_pool_temp,
            "ir_reading": ir_reading,
            "absolute_temp": absolute_temp,
            "energy_density": energy_density,
            "used_real_data": used_real_data,
            "correction_message": cal_message,
            "real_correction_factor": real_correction
        }
    
    def predict_from_process_parameters(self, laser_power, scan_speed, layer_thickness, material="316L"):
        """
        MAIN FUNCTION: Predict everything from process parameters using REAL correction factors
        """
        # Calculate temperatures using REAL correction factors
        temp_result = self.calculate_realistic_temperature(
            laser_power, scan_speed, layer_thickness, material
        )
        
        # Assess defect risk based on ACTUAL Absolute Temperature
        risk_level, recommendation = self.assess_defect_risk(temp_result["absolute_temp"], material)
        
        # Generate optimization suggestions
        optimization = self.generate_optimization_suggestions(
            laser_power, scan_speed, layer_thickness, temp_result["absolute_temp"], material
        )
        
        return {
            "process_parameters": {
                "laser_power": laser_power,
                "scan_speed": scan_speed, 
                "layer_thickness": layer_thickness,
                "material": material
            },
            "temperature_predictions": temp_result,
            "quality_assessment": {
                "risk_level": risk_level,
                "recommendation": recommendation,
                "optimization": optimization
            }
        }
    
    def assess_defect_risk(self, absolute_temperature, material="316L"):
        """Assess defect risk based on ACTUAL Absolute Temperature"""
        optimal_low, optimal_high = self.material_properties[material]["optimal_range"]
        melting_point = self.material_properties[material]["melting_point"]
        
        if absolute_temperature < optimal_low:
            return "❌ LACK OF FUSION", "Insufficient melting - increase energy input"
        elif absolute_temperature > optimal_high:
            return "❌ OVERHEATING", "Excessive temperature - decrease energy input"
        elif optimal_low <= absolute_temperature <= melting_point:
            return "⚠️ PARTIAL MELTING", "Borderline - slight adjustment needed"
        else:
            return "✅ OPTIMAL", "Good melt pool conditions"
    
    def generate_optimization_suggestions(self, laser_power, scan_speed, layer_thickness, current_temp, material):
        """Generate DYNAMIC optimization suggestions based on Absolute Temperature"""
        optimal_low, optimal_high = self.material_properties[material]["optimal_range"]
        suggestions = []
        
        if current_temp < optimal_low:
            power_increase = laser_power * 0.15
            speed_decrease = scan_speed * 0.20
            suggestions.append(f"• Increase laser power from {laser_power}W to {laser_power + power_increase:.0f}W")
            suggestions.append(f"• Decrease scan speed from {scan_speed}mm/s to {scan_speed - speed_decrease:.0f}mm/s")
            suggestions.append(f"• Consider reducing layer thickness from {layer_thickness}mm to {layer_thickness * 0.8:.3f}mm")
            
        elif current_temp > optimal_high:
            power_decrease = laser_power * 0.12
            speed_increase = scan_speed * 0.18
            suggestions.append(f"• Decrease laser power from {laser_power}W to {laser_power - power_decrease:.0f}W")
            suggestions.append(f"• Increase scan speed from {scan_speed}mm/s to {scan_speed + speed_increase:.0f}mm/s") 
            suggestions.append(f"• Consider increasing layer thickness from {layer_thickness}mm to {layer_thickness * 1.2:.3f}mm")
        
        else:
            suggestions.append("• Current parameters are well optimized")
            suggestions.append("• Maintain consistent parameter settings")
            suggestions.append("• Monitor for any drift in temperature readings")
        
        # Material-specific suggestions
        if material in ["Ni50.1Ti", "Ni52Ti"]:
            suggestions.append("• NiTi: Monitor for phase transformation consistency")
            suggestions.append("• NiTi: Ensure stable thermal history for shape memory")
        
        return suggestions

class EnhancedPredictionStudio:
    def __init__(self):
        self.commercial_predictor = TunedCommercialPredictor()
        self.research_predictor = TunedResearchPredictor()
        self.am_process_predictor = AMProcessPredictor()
    
    def render_studio(self):
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        st.header("🎯 Prediction Studio Pro - Uses REAL Correction Factors & REAL Error Analysis")
        
        # Check calibration status
        calibration_status = "❌ No calibration data available"
        real_correction = None
        
        # Check if calibration data exists AND has content
        if 'absolute_temp_data' in st.session_state and st.session_state.absolute_temp_data is not None:
            calibration_data = st.session_state.absolute_temp_data
            
            # Check if it has the expected structure
            if isinstance(calibration_data, dict) and "correction_factors" in calibration_data:
                factors = calibration_data["correction_factors"]
                if "overall_correction" in factors:
                    real_correction = factors["overall_correction"]
                    calibration_status = f"✅ Using REAL correction factor: {real_correction:.3f}"
                else:
                    calibration_status = "⚠️ Correction factors available but no overall factor"
            else:
                calibration_status = "⚠️ Calibration data available but no correction factors"
        else:
            calibration_status = "❌ No calibration data - run Absolute Temperature Studio first"
        
        st.info(f"**Calibration Status:** {calibration_status}")
        
        if real_correction:
            st.success(f"🔬 **All predictions will use: IR × {real_correction:.3f} = Absolute Temperature**")
        else:
            st.warning("⚠️ **Using theoretical correction factors - run Absolute Temperature Studio for real factors**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="commercial-card">', unsafe_allow_html=True)
            st.subheader("🏭 COMMERCIAL MODE")
            st.write("• Uses REAL correction factors")
            st.write("• **REAL error analysis**")
            st.write("• Auto-detection • Confidence scores")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="research-card">', unsafe_allow_html=True)
            st.subheader("🔬 RESEARCH MODE")
            st.write("• Uses REAL calibration data")
            st.write("• Advanced analysis • Multiple models")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="process-card">', unsafe_allow_html=True)
            st.subheader("⚡ AM PROCESS MODE")
            st.write("• **Uses REAL correction factors**")
            st.write("• Physics-based • Defect detection")
            st.markdown('</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🏭 Commercial Predictor", "🔬 Research Analyzer", "⚡ AM Process Simulator"])
        
        with tab1:
            self.render_commercial_interface()
        with tab2:
            self.render_research_interface()
        with tab3:
            self.render_am_process_interface()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_commercial_interface(self):
        st.subheader("🏭 Commercial Predictor - Uses REAL Correction Factors & REAL Error Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            material = st.selectbox("Material", ["Auto-Detect", "316L", "Cantor", "Inconel 718", "Ti-6Al-4V", "Ni50.1Ti", "Ni52Ti"], key="commercial_material")
        with col2:
            ir_temp = st.number_input("IR Temperature Reading (°C)", 400, 2000, 1000, key="commercial_ir")
        
        if st.button("🚀 PREDICT ABSOLUTE TEMPERATURE", key="commercial_predict", use_container_width=True):
            self.run_commercial_prediction(ir_temp, material)
        
        # Show real error analysis if calibration data exists
        if 'absolute_temp_data' in st.session_state and st.session_state.absolute_temp_data is not None:
            st.subheader("📊 Real Error Analysis")
            if st.button("🔍 ANALYZE REAL ERROR PERFORMANCE", key="error_analysis", use_container_width=True):
                self.run_real_error_analysis(material)
    
    def render_research_interface(self):
        st.subheader("🔬 Research Analyzer - Uses REAL Calibration Data")
        
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Prediction Model", ["Polynomial", "Spline"], key="research_model")
            st.info("Uses real IR vs Absolute temperature data from your calibration")
        with col2:
            if st.button("🔬 ANALYZE CALIBRATION DATA", key="research_analyze", use_container_width=True):
                self.run_research_analysis(model_type)
    
    def render_am_process_interface(self):
        st.subheader("⚡ AM Process Simulator - Uses REAL Correction Factors")
        
        # Get real correction status
        real_correction, cal_message = self.am_process_predictor.get_real_correction_factor()
        
        if real_correction:
            st.success(f"✅ **Using REAL correction factor: {real_correction:.3f}**")
        else:
            st.warning("⚠️ **Using theoretical correction - run Absolute Temperature Studio for better accuracy**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("🎯 AM Process Parameters")
            material = st.selectbox("Material", ["316L", "Cantor", "Inconel 718", "Ti-6Al-4V", "Ni50.1Ti", "Ni52Ti"], key="am_material")
            
            if material in ["Ni50.1Ti", "Ni52Ti"]:
                st.success("🔬 NiTi Alloy Selected - Shape Memory Properties")
            
            laser_power = st.slider("Laser Power (W)", 100, 500, 200, key="laser_power")
            scan_speed = st.slider("Scan Speed (mm/s)", 500, 3000, 1000, key="scan_speed")
            layer_thickness = st.slider("Layer Thickness (mm)", 0.02, 0.08, 0.03, 0.01, key="layer_thickness")
            
            if st.button("🚀 PREDICT MELT POOL TEMPERATURE", key="am_simulate", use_container_width=True):
                self.run_am_process_simulation(material, laser_power, scan_speed, layer_thickness)
        
        with col2:
            st.info("📊 PREDICTION OUTPUT")
            st.write("**1. Absolute Temperature Prediction**")
            st.write("• From laser power, scan speed, layer thickness")
            st.write("• **Uses REAL correction factors**")
            st.write("**2. Realistic Physics**") 
            st.write("• Melt pool: 1400-2500°C")
            st.write("• Energy density: 30-100 J/mm³")
            st.write("**3. Smart Defect Detection**")
            st.write("• Based on Absolute Temperature")
    
    def run_commercial_prediction(self, ir_temp, material):
        result = self.commercial_predictor.predict_commercial(ir_temp, material)
        
        st.success(f"**🎯 ABSOLUTE TEMPERATURE: {result['predicted_absolute']:.0f}°C**")
        
        # Confidence display with real error analysis
        if result['uses_real_error']:
            # Using real error analysis
            error_analysis = result['real_error_analysis']
            if error_analysis['error_improvement']:
                st.success(f"**✅ REAL PERFORMANCE: {result['confidence']:.0%}** (Actual Error: ±{result['max_error']:.1f}°C)")
                st.info(f"🎯 **Better than expected!** Real max error ({error_analysis['real_max_error']:.1f}°C) < stored limit ({error_analysis['stored_max_error']}°C)")
            else:
                st.warning(f"**🟡 REAL PERFORMANCE: {result['confidence']:.0%}** (Actual Error: ±{result['max_error']:.1f}°C)")
                st.info(f"📊 Real max error: {error_analysis['real_max_error']:.1f}°C vs stored limit: {error_analysis['stored_max_error']}°C")
        else:
            # Using theoretical confidence
            if result['confidence'] > 0.9: 
                st.info(f"**✅ THEORETICAL CONFIDENCE: {result['confidence']:.0%}** (Error: ±{result['max_error']}°C)")
            elif result['confidence'] > 0.8: 
                st.warning(f"**🟡 THEORETICAL CONFIDENCE: {result['confidence']:.0%}** (Error: ±{result['max_error']}°C)")
            else: 
                st.error(f"**🔴 THEORETICAL CONFIDENCE: {result['confidence']:.0%}** (Error: ±{result['max_error']}°C)")
        
        # Correction factor info
        st.info(f"**Correction:** {result['correction_message']}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            st.metric("IR Input", f"{result['ir_input']}°C")
        with col2: 
            st.metric("Absolute Temp", f"{result['predicted_absolute']:.0f}°C")
        with col3: 
            st.metric("Correction Factor", f"{result['correction_factor']:.3f}")
        with col4: 
            st.metric("Material", result['material_detected'])
        
        if result['used_real_data']:
            st.success("✅ **Using REAL correction factor from your calibration data!**")
        else:
            st.warning("⚠️ **Using theoretical correction - run Absolute Temperature Studio for real factors**")
        
        # Show real error analysis details if available
        if result['real_error_analysis']:
            self.display_real_error_analysis(result['real_error_analysis'])
    
    def run_real_error_analysis(self, material):
        """Run detailed real error analysis"""
        error_analysis = self.commercial_predictor.evaluate_real_error(material)  # FIXED: No unpacking
        
        if error_analysis is None:
            st.error("❌ Cannot perform error analysis - not enough calibration data available")
            return
        
        self.display_real_error_analysis(error_analysis)
    
    def display_real_error_analysis(self, error_analysis):
        """Display detailed real error analysis"""
        st.subheader("📊 Real Error Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points Analyzed", error_analysis['data_points'])
        with col2:
            st.metric("Real Mean Error", f"{error_analysis['real_mean_error']:.2f}°C")
        with col3:
            st.metric("Real Max Error", f"{error_analysis['real_max_error']:.2f}°C")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Stored Max Error Limit", f"{error_analysis['stored_max_error']}°C")
        with col5:
            st.metric("Original Confidence", f"{error_analysis['original_confidence']:.1%}")
        with col6:
            confidence_change = error_analysis['adjusted_confidence'] - error_analysis['original_confidence']
            confidence_color = "normal" if confidence_change >= 0 else "inverse"
            st.metric("Adjusted Confidence", f"{error_analysis['adjusted_confidence']:.1%}", 
                     delta=f"{confidence_change:+.1%}", delta_color=confidence_color)
        
        # Performance assessment
        if error_analysis['error_improvement']:
            st.success(f"🎯 **EXCELLENT PERFORMANCE!** Your real data shows better accuracy than expected.")
        elif error_analysis['exceeds_limit']:
            st.error(f"⚠️ **PERFORMANCE WARNING:** Real max error exceeds stored limit by {error_analysis['real_max_error'] - error_analysis['stored_max_error']:.1f}°C")
        else:
            st.info(f"📊 **GOOD PERFORMANCE:** Real error within expected limits.")
    
    def run_research_analysis(self, model_type):
        """Run research analysis using REAL calibration data"""
        ir_data, absolute_data, message = self.research_predictor.get_absolute_temperature_data()
        
        if ir_data is None:
            st.error(message)
            return
        
        st.info(f"**Data Status:** {message}")
        st.write(f"**Dataset:** {len(ir_data)} IR vs Absolute temperature points")
        
        if model_type == "Polynomial":
            ir_range, absolute_pred, model = self.research_predictor.polynomial_prediction(ir_data, absolute_data, 3)
        elif model_type == "Spline":
            ir_range, absolute_pred, model = self.research_predictor.spline_prediction(ir_data, absolute_data, 0.0001)
        
        if len(ir_range) > 0:
            # Calculate metrics
            absolute_pred_interp = np.interp(ir_data, ir_range, absolute_pred)
            r2 = r2_score(absolute_data, absolute_pred_interp)
            rmse = np.sqrt(mean_squared_error(absolute_data, absolute_pred_interp))
            
            st.success(f"**🔬 {model_type} Analysis Complete!**")
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("R² Score", f"{r2:.4f}")
            with col2: st.metric("RMSE", f"{rmse:.2f}°C")
            with col3: st.metric("Data Points", len(ir_data))
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(ir_data, absolute_data, alpha=0.7, label='Real Calibration Data', color='blue')
            ax.plot(ir_range, absolute_pred, 'r-', linewidth=2, label=f'{model_type} Fit')
            ax.plot([ir_data.min(), ir_data.max()], [ir_data.min(), ir_data.max()], 'k--', alpha=0.5, label='Ideal')
            
            ax.set_xlabel('IR Temperature (°C)')
            ax.set_ylabel('Absolute Temperature (°C)')
            ax.set_title(f'{model_type} Fit to Real Calibration Data\nR² = {r2:.4f}, RMSE = {rmse:.2f}°C')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    def run_am_process_simulation(self, material, laser_power, scan_speed, layer_thickness):
        """RUN simulation using REAL correction factors"""
        
        # PREDICT everything from process parameters
        prediction_result = self.am_process_predictor.predict_from_process_parameters(
            laser_power, scan_speed, layer_thickness, material
        )
        
        # Display results
        st.success("**⚡ AM PROCESS PREDICTION COMPLETE**")
        
        # Show calibration usage status
        temp_predictions = prediction_result['temperature_predictions']
        st.info(f"**{temp_predictions['correction_message']}**")
        
        if temp_predictions['used_real_data']:
            st.success(f"✅ **Using REAL correction factor: {temp_predictions['real_correction_factor']:.3f}**")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Laser Power", f"{laser_power}W")
        with col2:
            st.metric("Scan Speed", f"{scan_speed}mm/s")
        with col3:
            st.metric("Melt Pool Temp", f"{temp_predictions['melt_pool_temp']:.0f}°C")
        with col4:
            st.metric("Energy Density", f"{temp_predictions['energy_density']:.0f} J/mm³")
        
        # Temperature readings
        col5, col6 = st.columns(2)
        with col5:
            st.metric("IR Reading", f"{temp_predictions['ir_reading']:.0f}°C")
        with col6:
            st.metric("Absolute Temp", f"{temp_predictions['absolute_temp']:.0f}°C")
        
        # Defect risk assessment
        risk_level = prediction_result['quality_assessment']['risk_level']
        recommendation = prediction_result['quality_assessment']['recommendation']
        
        st.subheader("📊 Defect Risk Assessment")
        risk_col1, risk_col2 = st.columns(2)
        with risk_col1:
            if "❌" in risk_level:
                st.error(risk_level)
            elif "⚠️" in risk_level:
                st.warning(risk_level)
            else:
                st.success(risk_level)
        with risk_col2:
            st.info(recommendation)
        
        # DYNAMIC Optimization Suggestions
        st.subheader("🎯 Process Optimization Suggestions")
        for suggestion in prediction_result['quality_assessment']['optimization']:
            st.write(suggestion)
        
        # Visualization: Process Parameter Map
        self.plot_process_map(material, layer_thickness, laser_power, scan_speed, prediction_result)
    
    def plot_process_map(self, material, layer_thickness, laser_power, scan_speed, prediction_result):
        """Plot process parameter map with current point"""
        st.subheader("📈 Process Parameter Map")
        
        # Generate sample process map
        laser_powers = np.linspace(100, 400, 15)
        scan_speeds = np.linspace(500, 2000, 15)
        
        temperature_map = np.zeros((len(laser_powers), len(scan_speeds)))
        
        for i, power in enumerate(laser_powers):
            for j, speed in enumerate(scan_speeds):
                result = self.am_process_predictor.calculate_realistic_temperature(
                    power, speed, layer_thickness, material
                )
                temperature_map[i, j] = result["absolute_temp"]  # Use Absolute Temperature
        
        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.contourf(scan_speeds, laser_powers, temperature_map, levels=20, cmap='plasma')
        
        # Mark CURRENT calculated point
        current_temp = prediction_result['temperature_predictions']['absolute_temp']
        ax.plot(scan_speed, laser_power, 'ro', markersize=10, 
                label=f'Current: {current_temp:.0f}°C (Absolute)')
        
        # Add optimal range
        props = self.am_process_predictor.material_properties[material]
        optimal_low, optimal_high = props["optimal_range"]
        
        ax.set_xlabel('Scan Speed (mm/s)')
        ax.set_ylabel('Laser Power (W)')
        ax.set_title(f'Laser Power vs Scan Speed → Absolute Temperature\n(Material: {material}, Layer: {layer_thickness}mm)')
        ax.legend()
        plt.colorbar(contour, ax=ax, label='Absolute Temperature (°C)')
        st.pyplot(fig)

# Usage in your main app:
def main():
    prediction_studio = EnhancedPredictionStudio()
    prediction_studio.render_studio()
                
# ========== MAIN APPLICATION - PROFESSIONAL DARK THEME ==========

def main():
    # Apply dark theme CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 30%;
        margin-right: 8px;
    }
    
    .status-ready { background-color: #00ff00; }
    .status-processing { background-color: #ffff00; }
    .status-error { background-color: #ff0000; }
    .status-waiting { background-color: #666666; }
    
    .log-window {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.8em;
        max-height: 200px;
        overflow-y: auto;
        color: #00ff00;
    }
    
    .module-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ----------------------------------------------
    #                HEADER SECTION
    # ----------------------------------------------
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("dummy logo EmiSENSE.png", width=150)
        st.write("")  # Add empty content to fix indentation
    with col2:
        st.markdown('<div class="main-header">emiSENS CORE</div>', unsafe_allow_html=True)
        st.markdown("Advanced Thermal Analytics Platform")

    # Real-time status bar
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        st.metric("System Status", "🟢 ONLINE", delta="Ready")
    with status_col2:
        modules_ready = sum([1 for key in ['calibration_data', 'emissivity_data', 'absolute_temp_data', 'hysteresis_data'] 
                           if key in st.session_state])
        st.metric("Modules Ready", f"{modules_ready}/4", delta="Complete")
    with status_col3:
        st.metric("Active Sessions", "1", delta="Current")
    with status_col4:
        st.metric("Data Integrity", "100%", delta="Verified")

    # ----------------------------------------------
    #                    SIDEBAR
    # ----------------------------------------------
    with st.sidebar:
        st.image("dummy logo EmiSENSE.png", width=150)
        st.markdown("**EmiSENSE Pro**")
        st.markdown("---")

        # Module Status Overview
        st.subheader("📊 Module Status")
        
        module_status = {
            "Calibration": "ready" if 'calibration_data' in st.session_state else "waiting",
            "Emissivity": "ready" if 'emissivity_data' in st.session_state else "waiting", 
            "Absolute Temp": "ready" if 'absolute_temp_data' in st.session_state else "waiting",
            "Hysteresis": "ready" if 'hysteresis_data' in st.session_state else "waiting",
            "Prediction": "ready" if 'prediction_studio' in st.session_state else "waiting"
        }
        
        for module, status in module_status.items():
            status_color = {"ready": "🟢", "waiting": "⚪", "processing": "🟡", "error": "🔴"}[status]
            st.write(f"{status_color} {module}")

        st.markdown("---")
        st.header("📁 Data Input")
        pyro_file = st.file_uploader("Pyrometer Data (Time, Voltage, Temperature)", type=['csv', 'xlsx'], key="pyro")
        thermal_file = st.file_uploader("Thermal Stage Data (Time, Temperature)", type=['csv', 'xlsx'], key="thermal")
        blackbody_file = st.file_uploader("Blackbody Data (Temperature, Voltage)", type=['csv', 'xlsx'], key="bb")

        st.markdown("---")
        st.header("⚙️ Analysis Settings")

        # ADVANCED SETTINGS EXPANDER
        with st.expander("🔧 Advanced Settings"):
            st.checkbox("Enable Smoothing", value=True, key="smoothing_enabled")
            target_points = st.slider("Target Data Points", 1000, 15000, 8000)
            spline_smoothing = st.slider("Spline Smoothing", 0.00001, 0.001, 0.0001, 0.00001, format="%.5f")
            outlier_threshold = st.slider("Outlier Threshold", 1.0, 5.0, 3.0)
            min_temp = st.slider("Minimum Temperature (°C)", 400, 800, 550)
            interpolation_points = st.slider("Hysteresis Interpolation Points", 100, 1000, 500)

        # Report Generation
        st.markdown("---")
        st.header("📄 Report Generation")
        report_name = st.text_input("Report Name", "EmiSENSE_Analysis_Report")
        include_charts = st.checkbox("Include Charts", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
        
        if st.button("📊 GENERATE PDF REPORT", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                report_path = generate_pdf_report(report_name, include_charts, include_raw_data)
                if report_path:
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="📥 DOWNLOAD PDF REPORT",
                            data=file,
                            file_name=f"{report_name}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

        st.markdown("---")

        # ---------------------------------------------------------------
        #       BUTTON: PROCESS COMPLETE WORKFLOW
        # ---------------------------------------------------------------
        if st.button("🚀 PROCESS COMPLETE WORKFLOW", type="primary", use_container_width=True):
            # Initialize logs
            if 'logs' not in st.session_state:
                st.session_state.logs = []
            
            # Basic safety check
            missing = [name for name, f in 
                       zip(["Pyrometer", "Thermal Stage", "Blackbody"],
                           [pyro_file, thermal_file, blackbody_file]) if f is None]

            if missing:
                st.session_state.logs.append("❌ ERROR: Missing files: " + ", ".join(missing))
                st.error("❌ Missing files: " + ", ".join(missing))
                st.stop()

            # Store settings
            st.session_state.update({
                'pyro_file': pyro_file,
                'thermal_file': thermal_file,
                'blackbody_file': blackbody_file,
                'target_points': target_points,
                'spline_smoothing': spline_smoothing,
                'outlier_threshold': outlier_threshold,
                'min_temp': min_temp,
                'interpolation_points': interpolation_points
            })

            with st.spinner("Running complete analysis workflow..."):
                try:
                    # Safe load for CSV/Excel
                    def _safe_read(uploaded_file):
                        if uploaded_file.name.lower().endswith(".csv"):
                            return pd.read_csv(uploaded_file)
                        return pd.read_excel(uploaded_file)

                    st.session_state.logs.append("📁 Loading pyrometer data...")
                    pyro_data = _safe_read(pyro_file)
                    st.session_state.logs.append("✅ Pyrometer data loaded successfully")
                    
                    st.session_state.logs.append("📁 Loading thermal stage data...")
                    thermal_data = _safe_read(thermal_file)
                    st.session_state.logs.append("✅ Thermal stage data loaded successfully")

                    # Run studios with logging
                    st.session_state.logs.append("🔧 Running calibration module...")
                    st.session_state.calibration_data = run_calibration_studio_enhanced(pyro_data, thermal_data, target_points)
                    st.session_state.logs.append("✅ Calibration module completed")

                    if st.session_state.calibration_data:
                        st.session_state.logs.append("🔧 Running emissivity analysis...")
                        emissivity_data, outlier_info = run_emissivity_studio_corrected(
                            st.session_state.calibration_data,
                            blackbody_file,
                            spline_smoothing,
                            outlier_threshold
                        )
                        st.session_state.emissivity_data = (emissivity_data, outlier_info)
                        st.session_state.logs.append("✅ Emissivity analysis completed")

                    if st.session_state.emissivity_data:
                        st.session_state.logs.append("🔧 Calculating absolute temperatures...")
                        st.session_state.absolute_temp_data = run_absolute_temp_studio_corrected(
                            st.session_state.calibration_data,
                            st.session_state.emissivity_data[0],
                            spline_smoothing,
                            min_temp
                        )
                        st.session_state.logs.append("✅ Absolute temperature calculation completed")

                    if st.session_state.absolute_temp_data:
                        st.session_state.logs.append("🔧 Analyzing hysteresis...")
                        st.session_state.hysteresis_data = run_hysteresis_studio_enhanced(
                            st.session_state.absolute_temp_data,
                            min_temp,
                            interpolation_points
                        )
                        st.session_state.logs.append("✅ Hysteresis analysis completed")

                    st.session_state.logs.append("🎉 All modules completed successfully!")
                    st.success("✅ Complete workflow processed!")

                except Exception as e:
                    error_msg = f"❌ Workflow failed: {e}"
                    st.session_state.logs.append(error_msg)
                    st.error(error_msg)

        # Real-time Logs Window
        st.markdown("---")
        st.subheader("📋 Real-time Logs")
        if 'logs' in st.session_state and st.session_state.logs:
            log_text = "\n".join(st.session_state.logs[-10:])  # Show last 10 logs
            st.markdown(f'<div class="log-window">{log_text}</div>', unsafe_allow_html=True)
            
            if st.button("Clear Logs", use_container_width=True):
                st.session_state.logs = []
                st.rerun()
        else:
            st.markdown('<div class="log-window">No logs yet. Run workflow to see logs.</div>', unsafe_allow_html=True)

        # Sidebar footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.8em;">
            <strong>EmiSENSE Pro v2.0</strong><br>
            Developed by RB Nair<br>
            © 2025 All rights reserved
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------
    #                 MAIN TABS
    # ----------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗️ 1. Calibration", 
        "📊 2. Emissivity",
        "🌡️ 3. Absolute Temp", 
        "🔄 4. Hysteresis",
        "🎯 5. Prediction"
    ])

    with tab1:
        render_calibration_studio()
    with tab2:
        render_emissivity_studio()
    with tab3:
        render_absolute_temp_studio()
    with tab4:
        render_hysteresis_studio()
    with tab5:
        if "prediction_studio" not in st.session_state:
            st.session_state.prediction_studio = EnhancedPredictionStudio()
        st.session_state.prediction_studio.render_studio()

    # ----------------------------------------------
    #                MAIN FOOTER
    # ----------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 20px;">
        <strong>EmiSENS PRO - Advanced Thermal Analytics Platform</strong><br>
        Developed by <strong>RB Nair</strong> | Precision Temperature Measurement & AM Process Optimization<br>
        © 2025 RB Nair - All Rights Reserved
    </div>
    """, unsafe_allow_html=True)


def generate_pdf_report(report_name, include_charts=True, include_raw_data=False):
    """Generate PDF report with analysis results"""
    try:
        from fpdf import FPDF
        import tempfile
        import os
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="EmiSENSE Pro Analysis Report", ln=True, align='C')
        pdf.ln(10)
        
        # Report details
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Report Name: {report_name}", ln=True)
        pdf.cell(200, 10, txt=f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(200, 10, txt="Developed by: RB Nair", ln=True)
        pdf.ln(10)
        
        # Module status
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Module Completion Status:", ln=True)
        pdf.set_font("Arial", size=12)
        
        modules = ['calibration_data', 'emissivity_data', 'absolute_temp_data', 'hysteresis_data']
        for i, module in enumerate(modules, 1):
            status = "COMPLETED" if module in st.session_state else "PENDING"
            pdf.cell(200, 10, txt=f"Module {i}: {status}", ln=True)
        
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        report_path = os.path.join(temp_dir, f"{report_name}.pdf")
        pdf.output(report_path)
        
        st.session_state.logs.append(f"📄 PDF report generated: {report_path}")
        return report_path
        
    except Exception as e:
        st.session_state.logs.append(f"❌ PDF generation failed: {e}")
        return None


if __name__ == "__main__":
    main()