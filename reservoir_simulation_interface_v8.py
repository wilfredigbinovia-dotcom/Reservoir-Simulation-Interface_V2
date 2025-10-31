import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.set_page_config(page_title="Reservoir Simulation", layout="wide")

# -------------------------
# Helper: Corey relative permeability
# -------------------------
def corey_relperm(Sw, Swc, krw0, kro0, nw, no):
    Se = np.clip((Sw - Swc) / (1.0 - Swc), 0.0, 1.0)
    krw = krw0 * Se ** nw
    kro = kro0 * (1.0 - Se) ** no
    return krw, kro


# -------------------------
# Reservoir + Aquifer Coupled Simulator
# -------------------------
def run_simulation(params, relperm_table=None, Nr=60, tol_couple=1e-3, max_iter=20):
    # Unpack parameters
    J = float(params.get('J', 1e-5))
    N = float(params.get('N', 1e6))
    Bo = float(params.get('Bo', 1.2))
    Bw = float(params.get('Bw', 1.0))
    Rs = float(params.get('Rs', 600.0))
    Pi = float(params.get('Pi', 3000.0))
    Pwf = float(params.get('Pwf', 1000.0))
    ct = float(params.get('ct', 1e-5))
    mu_o = float(params.get('mu_o', 1.5))
    mu_w = float(params.get('mu_w', 0.5))

    # Corey defaults
    krw0 = float(params.get('krw0', 0.3))
    kro0 = float(params.get('kro0', 0.9))
    nw = float(params.get('nw', 3.0))
    no = float(params.get('no', 2.0))
    Swc = float(params.get('Swc', 0.2))

    # Aquifer params
    k_aq = float(params.get('k_aq', 100.0))
    h_aq = float(params.get('h_aq', 50.0))
    re = float(params.get('re', 2000.0))
    cw = float(params.get('cw', 1e-6))
    porosity = float(params.get('porosity', 0.25))
    rw = float(params.get('rw', 0.25))

    # Time discretization (days)
    time = np.linspace(0.0, 365.0, 50)
    dt_days = np.diff(time, prepend=0.0)

    # Initial states
    Np = 0.0
    Nw = 0.0
    Sw = Swc
    PV_reservoir = max(N * Bo / max(1e-12, (1.0 - Swc)), 1.0)

    # Relperm table
    if relperm_table is not None:
        relperm_table_sorted = relperm_table.sort_values('Sw')
        sw_array = relperm_table_sorted['Sw'].values
        krw_array = relperm_table_sorted['krw'].values
        kro_array = relperm_table_sorted['kro'].values
    else:
        sw_array = krw_array = kro_array = None

    # Aquifer radial grid
    r_edges = np.geomspace(rw, re, Nr + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)

    # Initial aquifer pressure (psi)
    P_aq = np.ones(Nr) * Pi

    # cumulative influx (STB)
    We_cum = 0.0

    # engineering constants
    conv_factor = 0.00708  # convert to STB/day from Darcy-based transmissivity

    # history arrays
    oil_rates = []
    water_rates = []
    gas_rates = []
    pressures = []
    Sw_history = []

    for step, dt in enumerate(dt_days):
        # coupling loop
        P_res_guess = Pi if len(pressures) == 0 else pressures[-1]
        iter_count = 0
        converged = False

        while (not converged) and (iter_count < max_iter):
            # relative permeability
            if relperm_table is not None:
                krw = float(np.interp(Sw, sw_array, krw_array, left=krw_array[0], right=krw_array[-1]))
                kro = float(np.interp(Sw, sw_array, kro_array, left=kro_array[0], right=kro_array[-1]))
            else:
                krw, kro = corey_relperm(Sw, Swc, krw0, kro0, nw, no)

            # mobilities & fractional flow
            lambda_w = krw / max(mu_w, 1e-6)
            lambda_o = kro / max(mu_o, 1e-6)
            denom = lambda_w + lambda_o
            fw = float(np.clip(lambda_w / denom if denom > 0 else 0.0, 1e-6, 0.999999))

            # production from P_res_guess
            q_oil = J * max(P_res_guess - Pwf, 0.0) / Bo
            q_oil = max(q_oil, 0.0)
            q_water = q_oil * fw / (1.0 - fw)
            q_water = min(q_water, 1e9)
            q_gas = (Rs * q_oil) / 1000.0

            produced_oil = q_oil * dt
            produced_water = q_water * dt

            Np_trial = Np + produced_oil

            # Build implicit aquifer matrix A * P_next = b
            A = np.zeros((Nr, Nr))
            b = np.zeros(Nr)

            for i in range(Nr):
                r_i = r_edges[i]
                r_e = r_edges[i + 1]
                V_cell_ft3 = np.pi * (r_e ** 2 - r_i ** 2) * h_aq
                pore_vol_ft3 = V_cell_ft3 * porosity
                storage_coef = pore_vol_ft3 * cw

                # left transmissivity
                if i == 0:
                    rface_left = r_edges[0]
                    T_left = conv_factor * k_aq * h_aq / max(mu_w, 1e-6) * (2.0 * np.pi * rface_left)
                    delta_r_left = r_centers[0] - r_edges[0]
                else:
                    rface_left = r_edges[i]
                    T_left = conv_factor * k_aq * h_aq / max(mu_w, 1e-6) * (2.0 * np.pi * rface_left)
                    delta_r_left = 0.5 * dr[i - 1] + 0.5 * dr[i]

                # right transmissivity
                if i == Nr - 1:
                    rface_right = r_edges[-1]
                    T_right = conv_factor * k_aq * h_aq / max(mu_w, 1e-6) * (2.0 * np.pi * rface_right)
                    delta_r_right = r_edges[-1] - r_centers[-1]
                else:
                    rface_right = r_edges[i + 1]
                    T_right = conv_factor * k_aq * h_aq / max(mu_w, 1e-6) * (2.0 * np.pi * rface_right)
                    delta_r_right = 0.5 * dr[i] + 0.5 * dr[i + 1]

                center = storage_coef / max(dt, 1e-12)
                left_coef = T_left / max(delta_r_left, 1e-12)
                right_coef = T_right / max(delta_r_right, 1e-12)
                center += left_coef + right_coef

                A[i, i] = center
                if i > 0:
                    A[i, i - 1] = -left_coef
                if i < Nr - 1:
                    A[i, i + 1] = -right_coef

                b_i = (storage_coef / max(dt, 1e-12)) * P_aq[i]
                if i == 0:
                    b_i += left_coef * P_res_guess
                if i == Nr - 1:
                    b_i += right_coef * Pi
                b[i] = b_i

            # solve for P_aq_next
            try:
                P_aq_next = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                A += np.eye(Nr) * 1e-12
                P_aq_next = np.linalg.solve(A, b)

            # inner-face flux
            delta_r_inner = r_centers[0] - r_edges[0]
            r_face_inner = r_edges[0]
            T_inner = conv_factor * k_aq * h_aq / max(mu_w, 1e-6) * (2.0 * np.pi * r_face_inner)
            Q_inner = - T_inner * (P_aq_next[0] - P_res_guess) / max(delta_r_inner, 1e-12)  # STB/day
            We_step = Q_inner * dt
            We_trial = We_cum + We_step

            # reservoir material balance -> new reservoir pressure
            P_res_new = Pi - ((Np_trial * Bo - We_trial * Bw) / (N * ct))
            P_res_new = max(P_res_new, Pwf)

            if abs(P_res_new - P_res_guess) < tol_couple:
                converged = True
            else:
                P_res_guess = P_res_new
                iter_count += 1
                P_aq = P_aq_next.copy()

        # Accept values after coupling
        P_aq = P_aq_next.copy()
        P_res = P_res_new
        We_cum += We_step

        # finalize production increments using P_res
        if relperm_table is not None:
            krw = float(np.interp(Sw, sw_array, krw_array, left=krw_array[0], right=krw_array[-1]))
            kro = float(np.interp(Sw, sw_array, kro_array, left=kro_array[0], right=kro_array[-1]))
        else:
            krw, kro = corey_relperm(Sw, Swc, krw0, kro0, nw, no)

        lambda_w = krw / max(mu_w, 1e-6)
        lambda_o = kro / max(mu_o, 1e-6)
        denom = lambda_w + lambda_o
        fw = float(np.clip(lambda_w / denom if denom > 0 else 0.0, 1e-6, 0.999999))

        q_oil = J * max(P_res - Pwf, 0.0) / Bo
        q_oil = max(q_oil, 0.0)
        q_water = q_oil * fw / (1.0 - fw)
        q_water = min(q_water, 1e9)
        q_gas = (Rs * q_oil) / 1000.0

        produced_oil = q_oil * dt
        produced_water = q_water * dt

        Np += produced_oil
        Nw += produced_water

        produced_water_rb = produced_water * Bw
        dSw = produced_water_rb / PV_reservoir
        Sw = float(np.clip(Sw + dSw, Swc, 0.95))

        # record history
        pressures.append(P_res)
        oil_rates.append(q_oil)
        water_rates.append(q_water)
        gas_rates.append(q_gas)
        Sw_history.append(Sw)

    return pd.DataFrame({
        'Time (days)': time,
        'ReservoirPressure_psi': pressures,
        'WaterSaturation': Sw_history,
        'OilRate_STB': oil_rates,
        'WaterRate_STB': water_rates,
        'GasRate_MSCF': gas_rates
    })


# -------------------------
# Streamlit Interface
# -------------------------
tab_pvt, tab_relperm, tab_aquifer, tab_history = st.tabs([
    "🧪 PVT Properties",
    "🌊 Relative Permeability",
    "💧 Aquifer Model",
    "⚙️ History Matching"
])

# PVT Tab
with tab_pvt:
    st.header("🧪 PVT Properties")
    col1, col2, col3 = st.columns(3)
    Bo = col1.number_input("Oil Formation Volume Factor (Bo, rb/stb)", value=1.2)
    mu_o = col2.number_input("Oil Viscosity (cp)", value=1.5)
    Rs = col3.number_input("Solution GOR (scf/stb)", value=600.0)
    col4, col5 = st.columns(2)
    Bw = col4.number_input("Water FVF (Bw, rb/stb)", value=1.0)
    mu_w = col5.number_input("Water Viscosity (cp)", value=0.5)
    st.markdown("> ✅ These PVT parameters will be passed into the simulation model.")

# Relative permeability tab
with tab_relperm:
    st.header("🌊 Relative Permeability Curves")
    st.write("Choose Corey parameters OR upload a table with columns (Sw, krw, kro).")
    input_method = st.radio("Input Method:", ["Corey Correlations", "Upload Table"])

    relperm_table = None
    if input_method == "Corey Correlations":
        col1, col2, col3 = st.columns(3)
        krw0 = col1.number_input("Endpoint krw0", value=0.3)
        kro0 = col2.number_input("Endpoint kro0", value=0.9)
        nw = col3.number_input("Water exponent (nw)", value=3.0)
        col4, col5 = st.columns(2)
        no = col4.number_input("Oil exponent (no)", value=2.0)
        Swc = col5.number_input("Connate Water Saturation (Swc)", value=0.2)
        st.markdown("> ✅ Corey parameters defined for internal kr generation.")
    else:
        uploaded_relperm = st.file_uploader("Upload RelPerm Table (Sw, krw, kro, optional)", type="csv")
        if uploaded_relperm:
            relperm_table = pd.read_csv(uploaded_relperm)
            st.dataframe(relperm_table.head())
        else:
            st.info("No table uploaded. You can still use Corey correlations.")

# Aquifer tab
with tab_aquifer:
    st.header("💧 Aquifer Model")
    col1, col2, col3 = st.columns(3)
    k_aq = col1.number_input("Aquifer Permeability (md)", value=100.0)
    h_aq = col2.number_input("Aquifer Thickness (ft)", value=50.0)
    re = col3.number_input("Outer Radius (ft)", value=2000.0)
    col4, col5 = st.columns(2)
    cw = col4.number_input("Water Compressibility (1/psi)", value=1e-6, format="%.1e")
    porosity = col5.number_input("Aquifer Porosity", value=0.25)
    rw = st.number_input("Wellbore Radius (rw, ft)", value=0.25, step=0.01, format="%.2f")
    st.markdown("> ✅ Aquifer parameters will be used in the implicit transient aquifer solver.")

# History matching tab
with tab_history:
    st.header("⚙️ History Matching & Optimization")

    # model params (note: include PVT and relperm params)
    model_params = {
        'J': st.number_input("Productivity Index (J, stb/day/psi)", value=1e-5, format="%.1e"),
        'N': st.number_input("OOIP (bbl)", value=1e6, format="%.1e"),
        'k_aq': k_aq,
        'h_aq': h_aq,
        're': re,
        'rw': rw,
        'Bo': Bo,
        'Bw': Bw,
        'mu_o': mu_o,
        'mu_w': mu_w,
        'Rs': Rs,
        'Pi': st.number_input("Initial Reservoir Pressure (psi)", value=3000.0),
        'Pwf': st.number_input("Bottom-hole Flowing Pressure (psi)", value=1000.0),
        'ct': st.number_input("Total Compressibility (1/psi)", value=1e-5, format="%.1e"),
        'cw': cw,
        'porosity': porosity
    }

    # add relperm/corey params if using Corey
    if relperm_table is None:
        try:
            model_params.update({
                'krw0': krw0,
                'kro0': kro0,
                'nw': nw,
                'no': no,
                'Swc': Swc
            })
        except NameError:
            pass

    st.subheader("📄 Historical Data")
    uploaded_file = st.file_uploader("Upload CSV (Days, OilRate_STB, WaterRate_STB, GasRate_MSCF)", type="csv")
    if uploaded_file is not None:
        hist_df_local = pd.read_csv(uploaded_file)
        st.dataframe(hist_df_local.head())
    else:
        st.info("No file uploaded — using synthetic history for demo.")
        days = np.linspace(0, 365, 50)
        hist_df_local = pd.DataFrame({
            'Days': days,
            'OilRate_STB': 200 * np.exp(-days / 200) + np.random.normal(0, 5, len(days)),
            'WaterRate_STB': 50 * (1 - np.exp(-days / 300)) + np.random.normal(0, 3, len(days)),
            'GasRate_MSCF': 1000 * np.exp(-days / 250) + np.random.normal(0, 50, len(days))
        })

    # weights for objective
    col_w1, col_w2, col_w3 = st.columns(3)
    w_oil = col_w1.number_input("Oil Weight", value=0.5, step=0.1)
    w_water = col_w2.number_input("Water Weight", value=0.3, step=0.1)
    w_gas = col_w3.number_input("Gas Weight", value=0.2, step=0.1)

    # ---- New: User-defined parameter bounds ----
    st.subheader("⚙️ Optimization Parameters and Bounds")
    col1, col2, col3 = st.columns(3)
    J_min = col1.number_input("Min J", value=1e-6, format="%.1e")
    J_max = col1.number_input("Max J", value=1e-3, format="%.1e")
    N_min = col2.number_input("Min N", value=1e5, format="%.1e")
    N_max = col2.number_input("Max N", value=1e7, format="%.1e")
    k_min = col3.number_input("Min k_aq", value=1.0)
    k_max = col3.number_input("Max k_aq", value=1e5)
    col4, col5 = st.columns(2)
    h_min = col4.number_input("Min h_aq", value=1.0)
    h_max = col4.number_input("Max h_aq", value=1e3)
    re_min = col5.number_input("Min re", value=100.0)
    re_max = col5.number_input("Max re", value=1e5)
    param_bounds = [(J_min, J_max), (N_min, N_max), (k_min, k_max), (h_min, h_max), (re_min, re_max)]

    # ---- Optional: Initial guess inputs ----
    st.markdown("Optionally set initial guesses for the optimizer:")
    colg1, colg2, colg3 = st.columns(3)
    J_init = colg1.number_input("Initial J", value=model_params['J'], format="%.1e")
    N_init = colg2.number_input("Initial N", value=model_params['N'], format="%.1e")
    k_init = colg3.number_input("Initial k_aq", value=model_params['k_aq'])
    colg4, colg5 = st.columns(2)
    h_init = colg4.number_input("Initial h_aq", value=model_params['h_aq'])
    re_init = colg5.number_input("Initial re", value=model_params['re'])
    initial_guess = [J_init, N_init, k_init, h_init, re_init]

    progress_bar = st.progress(0)
    phase_metrics = st.empty()
    current_rmse_text = st.empty()
    progress_data = []

    # Objective function for optimizer
    def objective(params_arr):
        Jv, Nv, k_aq_v, h_aq_v, re_v = params_arr
        # Update model params (use floats)
        model_params['J'] = float(Jv)
        model_params['N'] = float(Nv)
        model_params['k_aq'] = float(k_aq_v)
        model_params['h_aq'] = float(h_aq_v)
        model_params['re'] = float(re_v)

        sim_df = run_simulation(model_params, relperm_table=relperm_table)
        sim_time = sim_df['Time (days)'].values
        hist_days = hist_df_local['Days'].values

        sim_Qo = np.interp(hist_days, sim_time, sim_df['OilRate_STB'])
        sim_Qw = np.interp(hist_days, sim_time, sim_df['WaterRate_STB'])
        sim_Qg = np.interp(hist_days, sim_time, sim_df['GasRate_MSCF'])

        hist_oil = hist_df_local['OilRate_STB'].values
        hist_water = hist_df_local['WaterRate_STB'].values
        hist_gas = hist_df_local['GasRate_MSCF'].values

        rmse_oil = np.sqrt(np.mean((sim_Qo - hist_oil) ** 2))
        rmse_water = np.sqrt(np.mean((sim_Qw - hist_water) ** 2))
        rmse_gas = np.sqrt(np.mean((sim_Qg - hist_gas) ** 2))

        # Weighted RMSE combination (root of weighted sum of squares)
        rmse_total = np.sqrt(w_oil * rmse_oil ** 2 + w_water * rmse_water ** 2 + w_gas * rmse_gas ** 2)

        progress_data.append({
            'RMSE_Oil': rmse_oil,
            'RMSE_Water': rmse_water,
            'RMSE_Gas': rmse_gas,
            'RMSE_Total': rmse_total
        })

        i = len(progress_data)
        # update progress bar (normalize by an arbitrary cap, e.g., 50 iters)
        progress_bar.progress(min(i / 50, 1.0))
        phase_metrics.markdown(f"**Iteration {i}** — Oil RMSE: {rmse_oil:.2f}, Water RMSE: {rmse_water:.2f}, Gas RMSE: {rmse_gas:.2f}")
        current_rmse_text.write(f"**Total RMSE:** {rmse_total:.4f}")

        return float(rmse_total)

    # Run optimization button
    if st.button("Run Optimization"):
        with st.spinner("Running optimization... (this may take a while)"):
            # Use L-BFGS-B so we respect bounds
            result = minimize(objective, initial_guess, bounds=param_bounds, method='L-BFGS-B', options={'maxiter': 50})
        st.success("✅ Optimization Complete!")
        st.write("Optimized Parameters:", result.x)
        st.write("Final RMSE:", result.fun)

        df_progress = pd.DataFrame(progress_data)
        if not df_progress.empty:
            st.subheader("Optimization Progress")
            st.line_chart(df_progress[['RMSE_Oil', 'RMSE_Water', 'RMSE_Gas', 'RMSE_Total']])

        # Final simulation with best-fit parameters
        model_params['J'], model_params['N'], model_params['k_aq'], model_params['h_aq'], model_params['re'] = result.x
        sim_final = run_simulation(model_params, relperm_table=relperm_table)

        # Merge simulated + historical data
        hist_days = hist_df_local['Days'].values
        merged_df = pd.DataFrame({
            'Days': hist_days,
            'Historical_OilRate': hist_df_local['OilRate_STB'],
            'Simulated_OilRate': np.interp(hist_days, sim_final['Time (days)'], sim_final['OilRate_STB']),
            'Historical_WaterRate': hist_df_local['WaterRate_STB'],
            'Simulated_WaterRate': np.interp(hist_days, sim_final['Time (days)'], sim_final['WaterRate_STB']),
            'Historical_GasRate': hist_df_local['GasRate_MSCF'],
            'Simulated_GasRate': np.interp(hist_days, sim_final['Time (days)'], sim_final['GasRate_MSCF'])
        })

        st.subheader("📊 Merged Historical & Simulated Data (first rows)")
        st.dataframe(merged_df.head())

        # Export CSV
        csv = merged_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Merged Simulated Data (CSV)",
            data=csv,
            file_name="Merged_History_Simulation.csv",
            mime="text/csv"
        )

        # --- Plot results: Separate subplots for Oil, Water, Gas ---
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # --- Oil subplot ---
        axes[0].plot(hist_df_local['Days'], hist_df_local['OilRate_STB'], 'ro', label='Oil Rate (Hist)')
        axes[0].plot(sim_final['Time (days)'], sim_final['OilRate_STB'], 'r-', label='Oil Rate (Sim)')
        axes[0].set_ylabel("Oil Rate (STB/day)")
        axes[0].legend()
        axes[0].grid(True)

        # --- Water subplot ---
        axes[1].plot(hist_df_local['Days'], hist_df_local['WaterRate_STB'], 'bo', label='Water Rate (Hist)')
        axes[1].plot(sim_final['Time (days)'], sim_final['WaterRate_STB'], 'b-', label='Water Rate (Sim)')
        axes[1].set_ylabel("Water Rate (STB/day)")
        axes[1].legend()
        axes[1].grid(True)

        # --- Gas subplot ---
        axes[2].plot(hist_df_local['Days'], hist_df_local['GasRate_MSCF'], 'go', label='Gas Rate (Hist)')
        axes[2].plot(sim_final['Time (days)'], sim_final['GasRate_MSCF'], 'g-', label='Gas Rate (Sim)')
        axes[2].set_ylabel("Gas Rate (MSCF/day)")
        axes[2].set_xlabel("Time (days)")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        st.pyplot(fig)