#!/usr/bin/env python3
"""db_analyzer.py

Analyze a SQLite database, generate visualizations and an HTML report, and optionally send via email.

Usage:
  python db_analyzer.py --db sales_agent.db --email recipient@example.com [--send]

Email credentials are read from environment variables:
  SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD

"""
import os
import sys
import argparse
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


OUTPUT_DIR = "output"


def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def discover_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    return tables


def table_info(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    cols = cur.fetchall()
    # cols: cid, name, type, notnull, dflt_value, pk
    return cols


def read_table(conn, table, limit=None):
    q = f"SELECT * FROM '{table}'"
    if limit:
        q += f" LIMIT {limit}"
    return pd.read_sql_query(q, conn)


def analyze_table(conn, table):
    df = read_table(conn, table)
    info = {}
    info['row_count'] = len(df)
    info['columns'] = list(df.columns)
    info['dtypes'] = df.dtypes.astype(str).to_dict()
    info['nulls'] = df.isnull().sum().to_dict()
    # duplicates by all columns
    try:
        info['duplicate_rows'] = int(df.duplicated().sum())
    except Exception:
        info['duplicate_rows'] = None
    # basic stats for numeric
    try:
        info['describe'] = df.describe(include='all').to_dict()
    except Exception:
        info['describe'] = {}
    info['sample'] = df.head(5).to_dict(orient='records')
    info['df'] = df
    return info


def save_chart(fig, name, dpi=200):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path


def chart_table_row_counts(table_infos):
    sns.set_theme(style='whitegrid')
    names = [t for t in table_infos.keys()]
    counts = [table_infos[t]['row_count'] for t in names]
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=counts, y=names, palette='Blues_d', ax=ax)
    ax.set_title('Row Counts per Table')
    ax.set_xlabel('Rows')
    ax.set_ylabel('Table')
    fig.tight_layout()
    return save_chart(fig, 'chart_table_counts.png', dpi=200)


def find_time_column(df):
    # heuristics: look for columns with 'date' or 'time' in name
    for c in df.columns:
        low = c.lower()
        if 'date' in low or 'time' in low or 'day' in low:
            return c
    return None


def chart_time_trend(table_infos):
    # attempt to find a time series from any table with a numeric value column
    for t, info in table_infos.items():
        df = info['df']
        if df.empty:
            continue
        time_col = find_time_column(df)
        # find a numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if time_col and numeric_cols:
            try:
                df2 = df.copy()
                df2[time_col] = pd.to_datetime(df2[time_col], errors='coerce')
                df2 = df2.dropna(subset=[time_col])
                if df2.empty:
                    continue
                agg = df2.set_index(time_col)[numeric_cols].resample('M').sum()
                # nicer formatting: larger figure, rotated xlabels, better legend placement
                fig, ax = plt.subplots(figsize=(12,7))
                agg.plot(ax=ax, linewidth=2)
                ax.set_title(f'Time Series ({t})', fontsize=14)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Aggregated Value', fontsize=12)
                # format dates on x-axis
                try:
                    import matplotlib.dates as mdates
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
                except Exception:
                    pass
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_horizontalalignment('right')
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
                fig.tight_layout()
                return save_chart(fig, 'chart_time_trend.png', dpi=200)
            except Exception:
                continue
    # fallback: make a dummy line chart of cumulative table rows
    names = list(table_infos.keys())
    counts = [table_infos[n]['row_count'] for n in names]
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(range(len(counts)), counts, marker='o', linewidth=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title('Table Row Counts (fallback trend)', fontsize=13)
    ax.set_ylabel('Rows')
    fig.tight_layout()
    return save_chart(fig, 'chart_time_trend.png', dpi=200)


def chart_correlation(table_infos):
    # pick largest table and plot correlation heatmap of numeric columns
    largest = max(table_infos.keys(), key=lambda t: table_infos[t]['row_count'] if table_infos[t]['row_count'] is not None else 0)
    df = table_infos[largest]['df']
    num = df.select_dtypes(include=['number'])
    if num.shape[1] >= 2:
        corr = num.corr()
        # choose figsize proportional to number of columns to avoid cramped labels
        n = corr.shape[0]
        # width/height per cell (inches)
        cell_size = 0.5
        fig_w = max(8, n * cell_size + 3)
        fig_h = max(6, n * cell_size + 3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        # mask upper triangle for clarity
        try:
            import numpy as np
            mask = np.triu(np.ones_like(corr, dtype=bool))
        except Exception:
            mask = None
        # adapt annotation size to matrix size
        if n <= 8:
            annot_size = 10
        elif n <= 15:
            annot_size = 8
        else:
            annot_size = 6
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='vlag',
            mask=mask,
            ax=ax,
            cbar_kws={'shrink':0.6},
            annot_kws={'size': annot_size},
            linewidths=0.4,
            square=False,
        )
        ax.set_title(f'Correlation Heatmap ({largest})', fontsize=14)
        # rotate tick labels and align
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=max(8, annot_size))
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=max(8, annot_size))
        # expand layout margins to give space for rotated labels and colorbar
        try:
            fig.subplots_adjust(left=0.2, right=0.85, top=0.92, bottom=0.25)
        except Exception:
            pass
        fig.tight_layout()
        return save_chart(fig, 'chart_correlation.png', dpi=250)
    else:
        # fallback scatter between first two numeric columns across tables
        pairs = []
        for t, info in table_infos.items():
            num = info['df'].select_dtypes(include=['number'])
            if num.shape[1] >= 2:
                a,b = num.columns[:2]
                fig, ax = plt.subplots(figsize=(10,6))
                ax.scatter(num[a], num[b], alpha=0.6, edgecolors='w', s=40)
                ax.set_xlabel(a, fontsize=11)
                ax.set_ylabel(b, fontsize=11)
                ax.set_title(f'Scatter: {a} vs {b} ({t})', fontsize=13)
                fig.tight_layout()
                return save_chart(fig, 'chart_correlation.png', dpi=200)
        # last fallback: empty placeholder
        fig, ax = plt.subplots(figsize=(8,5))
        ax.text(0.5,0.5,'No numeric data for correlation',ha='center',va='center', fontsize=12)
        ax.axis('off')
        fig.tight_layout()
        return save_chart(fig, 'chart_correlation.png', dpi=200)


def build_report(db_path, table_infos, charts):
    total_tables = len(table_infos)
    total_records = sum([table_infos[t]['row_count'] for t in table_infos.keys() if table_infos[t]['row_count'] is not None])
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    insights = []
    # simple insights: largest table, most nulls column
    if table_infos:
        largest = max(table_infos.keys(), key=lambda t: table_infos[t]['row_count'] or 0)
        insights.append(f'Largest table: {largest} with {table_infos[largest]["row_count"]} rows')
        # find column with most nulls
        max_null = (None, 0, None)
        for t, info in table_infos.items():
            for c, n in info['nulls'].items():
                if n and n > max_null[1]:
                    max_null = (t, n, c)
        if max_null[0]:
            insights.append(f'Column with most nulls: {max_null[2]} in table {max_null[0]} ({max_null[1]} nulls)')
    if len(insights) < 3:
        insights.append('General: Data appears consistent; consider deeper domain-specific checks.')

    # build HTML
    html = []
    html.append('<html><head><meta charset="utf-8"><title>Database Analysis Report</title></head><body>')
    html.append(f'<h1>Database Analysis Report</h1>')
    html.append(f'<p><strong>DB file:</strong> {db_path}<br/><strong>Analysis Date:</strong> {timestamp}</p>')
    html.append('<h2>Database Summary</h2>')
    html.append(f'<ul><li>Total Tables: {total_tables}</li><li>Total Records: {total_records}</li></ul>')
    html.append('<h2>Key Insights</h2>')
    html.append('<ol>')
    for ins in insights[:5]:
        html.append(f'<li>{ins}</li>')
    html.append('</ol>')

    html.append('<h2>Per-table Overview</h2>')
    for t, info in table_infos.items():
        html.append(f'<h3>{t} ({info["row_count"]} rows)</h3>')
        html.append('<table border="1" cellpadding="4" cellspacing="0">')
        html.append('<tr><th>Column</th><th>Type</th><th>Nulls</th></tr>')
        for col in info['dtypes'].keys():
            html.append(f'<tr><td>{col}</td><td>{info["dtypes"][col]}</td><td>{info["nulls"].get(col,0)}</td></tr>')
        html.append('</table>')

    html.append('<h2>Visualizations</h2>')
    for c in charts:
        html.append(f'<h3>{os.path.basename(c)}</h3>')
        # embed image as relative link
        html.append(f'<img src="{c}" style="max-width:100%;height:auto;"/>')

    html.append('<p>Generated by AI CODEFIX 2025 - Team</p>')
    html.append('</body></html>')

    out_path = os.path.join(OUTPUT_DIR, 'report.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    # return report path and structured summary for email body
    summary = {
        'total_tables': total_tables,
        'total_records': total_records,
        'timestamp': timestamp,
        'insights': insights,
    }
    return out_path, summary


def html_to_pdf(html_path):
    """Attempt to convert HTML report to PDF using pdfkit (wkhtmltopdf).
    Returns path to PDF if successful, otherwise None."""
    try:
        import pdfkit
    except Exception:
        return None
    pdf_path = os.path.join(OUTPUT_DIR, 'report.pdf')
    try:
        # try default conversion; pdfkit will raise if wkhtmltopdf missing
        pdfkit.from_file(html_path, pdf_path)
        return pdf_path
    except Exception:
        return None


def reportlab_pdf(db_path, table_infos, charts):
    """Generate a simple PDF using reportlab (no system deps).
    Writes `output/report.pdf` and returns its path."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.units import inch
    except Exception:
        return None

    pdf_path = os.path.join(OUTPUT_DIR, 'report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 50

    # Header
    c.setFont('Helvetica-Bold', 16)
    c.drawString(margin, height - margin, 'Database Analysis Report')
    c.setFont('Helvetica', 10)
    c.drawString(margin, height - margin - 18, f'DB file: {db_path}')
    c.drawString(margin, height - margin - 32, f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')

    # Key insights (simple)
    y = height - margin - 60
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, 'Key Insights:')
    y -= 16
    c.setFont('Helvetica', 10)
    insights = []
    if table_infos:
        largest = max(table_infos.keys(), key=lambda t: table_infos[t]['row_count'] or 0)
        insights.append(f'Largest table: {largest} ({table_infos[largest]["row_count"]} rows)')
        max_null = (None, 0, None)
        for t, info in table_infos.items():
            for col, n in info['nulls'].items():
                if n and n > max_null[1]:
                    max_null = (t, n, col)
        if max_null[0]:
            insights.append(f'Column with most nulls: {max_null[2]} in {max_null[0]} ({max_null[1]} nulls)')
    if len(insights) == 0:
        insights.append('No prominent insights detected.')

    for ins in insights:
        c.drawString(margin + 10, y, f'- {ins}')
        y -= 14
        if y < margin + 120:
            c.showPage()
            y = height - margin

    # Per-table brief
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, 'Per-table Summary:')
    y -= 18
    c.setFont('Helvetica', 9)
    for t, info in table_infos.items():
        line = f'{t}: {info["row_count"]} rows, cols={len(info["columns"])}'
        c.drawString(margin + 6, y, line)
        y -= 12
        if y < margin + 120:
            c.showPage()
            y = height - margin

    # Add charts each on its own page
    for chart in charts:
        try:
            c.showPage()
            img = ImageReader(chart)
            iw, ih = img.getSize()
            # scale to page width minus margins
            max_w = width - 2 * margin
            scale = min(1.0, max_w / iw)
            draw_w = iw * scale
            draw_h = ih * scale
            x = (width - draw_w) / 2
            y_img = (height - draw_h) / 2
            c.drawImage(img, x, y_img, width=draw_w, height=draw_h)
        except Exception:
            continue

    try:
        c.save()
        return pdf_path
    except Exception:
        return None


def send_email(recipients, subject, body, attachments):
    server = os.environ.get('SMTP_SERVER')
    port = os.environ.get('SMTP_PORT')
    sender = os.environ.get('SENDER_EMAIL')
    password = os.environ.get('SENDER_PASSWORD')
    if not (server and port and sender and password):
        print('SMTP credentials not fully configured in environment; skipping send.')
        return False

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients if isinstance(recipients, list) else [recipients])
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    for path in attachments:
        with open(path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
        msg.attach(part)

    try:
        smtp = smtplib.SMTP(server, int(port), timeout=30)
        smtp.starttls()
        smtp.login(sender, password)
        smtp.send_message(msg)
        smtp.quit()
        print('Email sent successfully')
        return True
    except Exception as e:
        print('Failed to send email:', e)
        return False


def main():
    parser = argparse.ArgumentParser(description='Database Insights Agent')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--email', required=False, help='Recipient email address')
    parser.add_argument('--send', action='store_true', help='Attempt to send email (requires SMTP env vars)')
    args = parser.parse_args()

    db_path = args.db

    if not os.path.exists(db_path):
        print(f'Database file not found: {db_path}')
        sys.exit(2)

    ensure_output()

    conn = sqlite3.connect(db_path)
    tables = discover_tables(conn)
    print('Discovered tables:', tables)

    table_infos = {}
    for t in tables:
        try:
            info = analyze_table(conn, t)
            table_infos[t] = info
        except Exception as e:
            print(f'Error analyzing {t}:', e)

    # generate charts
    charts = []
    try:
        c1 = chart_table_row_counts(table_infos)
        charts.append(c1)
    except Exception as e:
        print('Failed chart 1:', e)
    try:
        c2 = chart_time_trend(table_infos)
        charts.append(c2)
    except Exception as e:
        print('Failed chart 2:', e)
    try:
        c3 = chart_correlation(table_infos)
        charts.append(c3)
    except Exception as e:
        print('Failed chart 3:', e)

    report_path, summary = build_report(db_path, table_infos, charts)
    print('Report generated at', report_path)

    # attempt to make a PDF copy of the report (optional)
    pdf_path = None
    try:
        pdf_path = html_to_pdf(report_path)
        if pdf_path:
            print('PDF report generated at', pdf_path)
        else:
            # fall back to reportlab-based PDF generation (no system deps)
            pdf_path = reportlab_pdf(db_path, table_infos, charts)
            if pdf_path:
                print('PDF report generated with reportlab at', pdf_path)
            else:
                print('PDF conversion not available (pdfkit missing) and reportlab failed or missing')
    except Exception:
        pdf_path = None
        try:
            pdf_path = reportlab_pdf(db_path, table_infos, charts)
            if pdf_path:
                print('PDF report generated with reportlab at', pdf_path)
        except Exception:
            print('PDF conversion failed or not available')

    if args.send and args.email:
        subject = 'Database Analysis Report - AI CODEFIX Team'
        # format body according to README template
        team_name = os.environ.get('TEAM_NAME', 'AI CODEFIX Team')
        body_lines = []
        body_lines.append(f'Dear Recipient,')
        body_lines.append('')
        body_lines.append('Please find the automated database analysis report below.')
        body_lines.append('')
        body_lines.append('=== DATABASE SUMMARY ===')
        body_lines.append(f"- Total Tables: {summary.get('total_tables')}")
        body_lines.append(f"- Total Records: {summary.get('total_records')}")
        body_lines.append(f"- Analysis Date: {summary.get('timestamp')}")
        body_lines.append('')
        body_lines.append('=== KEY INSIGHTS ===')
        ins = summary.get('insights', [])
        for i in range(3):
            if i < len(ins):
                body_lines.append(f"{i+1}. {ins[i]}")
            else:
                body_lines.append(f"{i+1}. None")
        body_lines.append('')
        body_lines.append('PDF( report with charts  Proffesional  business report with visualization and chart )')
        body_lines.append('[Attached: chart_table_counts.png, chart_time_trend.png, chart_correlation.png]')
        body_lines.append('')
        body_lines.append(f'Best regards,')
        body_lines.append(f'{team_name}')
        body_lines.append('AI CODEFIX 2025')
        body = '\n'.join(body_lines)

        attachments = charts + [report_path]
        # attach PDF if available
        try:
            if 'pdf_path' in locals() and pdf_path:
                attachments.append(pdf_path)
        except Exception:
            pass

        send_email(args.email, subject, body, attachments)
    else:
        if args.send and not args.email:
            print('--send requested but no --email given; skipping send')


if __name__ == '__main__':
    main()
