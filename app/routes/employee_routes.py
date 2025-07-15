# coorporate_app/app/routes/employee_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user, logout_user
from app.models import Personnels, Personnel_Entries, Personnel_Images, Tracking # Import models
from app import db # Import instance db
from app.utils.decorators import employee_required # Decorators
from app.utils.config_variables import get_personnel_folder_path
from datetime import datetime, date, timedelta
from sqlalchemy import func, cast, Date, text # Untuk fungsi database
import os
import shutil # Untuk operasi file/folder
from werkzeug.utils import secure_filename # Untuk upload file

bp = Blueprint('employee', __name__, template_folder='../templates/employee_panel')

def get_relative_image_path(absolute_path):
    if not absolute_path:
        return None
    try:
        # Base folder static (ubah sesuai struktur folder kamu)
        static_root_on_server = os.path.normpath(os.path.join(current_app.root_path, 'static'))
        absolute_path_norm = os.path.normpath(absolute_path)

        # Cek apakah path gambar berada di dalam static folder
        if os.path.commonpath([absolute_path_norm, static_root_on_server]) == static_root_on_server:
            relative_path = os.path.relpath(absolute_path_norm, static_root_on_server)
            return relative_path.replace("\\", "/")  # Gunakan format URL
        else:
            current_app.logger.warning(
                f"Tidak bisa mengubah path gambar absolut ke relatif: {absolute_path} menggunakan static root '{static_root_on_server}'")
            return None
    except Exception as e:
        current_app.logger.error(f"Error di get_relative_image_path: {e}")
        return None

def _get_employee_presence_data(target_date=None, personnel_id=None):
    sql = '''
        SELECT 
            p.id AS personnel_id,
            p.name,
            MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END) AS attended_time,
            MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END) AS leaving_time,
            CASE 
                WHEN EXISTS (
                    SELECT 1 
                    FROM personnel_entries AS sub 
                    WHERE sub.personnel_id = p.id 
                    AND sub.presence_status = 'LEAVE'
                ) THEN 'LEAVING'
                ELSE MAX(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.presence_status END)
            END AS latest_status,
            TIMESTAMPDIFF(HOUR, 
                MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
            ) AS work_hours,
            CASE 
                WHEN TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                    MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
                ) > 8 THEN CONCAT('Overtime ', TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                    MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
                ) - 8, ' hours')
                WHEN TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                    MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
                ) < 8 THEN CONCAT('Less time ', 8 - TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                    MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
                ), ' hours')
                ELSE 'Standard Time'
            END AS notes,
            (SELECT d2.image 
             FROM personnel_entries AS d2 
             WHERE d2.personnel_id = p.id 
             AND d2.presence_status IN ('ONTIME', 'LATE')
             ORDER BY d2.timestamp DESC 
             LIMIT 1
            ) AS attendance_image,
            (SELECT d3.image 
             FROM personnel_entries AS d3 
             WHERE d3.personnel_id = p.id 
             AND d3.presence_status = 'LEAVE'
             ORDER BY d3.timestamp DESC 
             LIMIT 1
            ) AS leaving_image
        FROM 
            personnel_entries AS d
        JOIN 
            personnels AS p ON p.id = d.personnel_id
    '''

    params = {}
    where_clauses = []

    if target_date:
        where_clauses.append("DATE(d.timestamp) = :target_date")
        params['target_date'] = target_date
    if personnel_id:
        where_clauses.append("p.id = :personnel_id")
        params['personnel_id'] = personnel_id

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sql += " GROUP BY p.id"

    result = db.session.execute(text(sql), params)
    entries = result.fetchall()

    presence_data = []
    for entry in entries:
        attended_time = entry.attended_time.strftime('%H:%M:%S') if entry.attended_time else '-'
        leaving_time = entry.leaving_time.strftime('%H:%M:%S') if entry.leaving_time else '-'

        # Gunakan fungsi konversi path
        attendance_image_path = get_relative_image_path(entry.attendance_image) if entry.attendance_image else 'img/no_image.png'
        leaving_image_path = get_relative_image_path(entry.leaving_image) if entry.leaving_image else 'img/no_image.png'

        presence_data.append({
            'id': entry.personnel_id,
            'name': entry.name,
            'date': entry.attended_time.date().isoformat() if entry.attended_time else '-',
            'attended': attended_time,
            'leave': leaving_time,
            'status': entry.latest_status,
            'work_hours': entry.work_hours if entry.work_hours is not None else 'Still Working',
            'notes': entry.notes if entry.notes is not None else 'No notes',
            'attendance_image_path': attendance_image_path,
            'leaving_image_path': leaving_image_path,
        })

    return presence_data

def get_employee_violation_summary(personnel_id, filter_date):
    """
    Mengembalikan dict: { 'no_lanyard_idcard': int, 'no_tie': int, 'total': int }
    """
    violation_map = {
        'Tidak Memakai Dasi': 'no_tie',
        'no-tie': 'no_tie'
    }
    summary = { 'no_tie': 0, 'total': 0}
    if not personnel_id or not filter_date:
        return summary

    query = (
        db.session.query(
            Tracking.detected_class,
            func.count(Tracking.id).label('count')
        )
        .filter(
            Tracking.personnel_id == personnel_id,
            func.date(Tracking.timestamp) == filter_date
        )
        .group_by(Tracking.detected_class)
    )
    total = 0
    for row in query:
        key = violation_map.get(row.detected_class)
        if key:
            summary[key] += row.count
            total += row.count
    summary['total'] = total
    return summary

@bp.route('/dashboard')
@login_required
@employee_required
def dashboard():
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        flash("Your account is not linked to personnel data. Please contact admin.", "danger")
        logout_user()
        return redirect(url_for('auth.login'))

    # Ambil tanggal filter dari query string, default hari ini
    filter_date_str = request.args.get('filter_date')
    today = date.today()
    if filter_date_str:
        try:
            filter_date = datetime.strptime(filter_date_str, '%Y-%m-%d').date()
        except ValueError:
            filter_date = today
            filter_date_str = today.strftime('%Y-%m-%d')
    else:
        filter_date = today
        filter_date_str = today.strftime('%Y-%m-%d')

    # Hitung total pelanggaran dan per jenis untuk pegawai ini pada tanggal filter
    violation_query = (
        db.session.query(
            Tracking.detected_class,
            func.count(Tracking.id).label('count')
        )
        .filter(
            Tracking.personnel_id == personnel.id,
            func.date(Tracking.timestamp) == filter_date
        )
        .group_by(Tracking.detected_class)
    )
    violation_summary = {'total': 0}
    for row in violation_query:
        violation_summary[row.detected_class] = row.count
        violation_summary['total'] += row.count

    # Data grafik: 7 hari terakhir
    days_range = 7
    chart_labels = []
    chart_data = []
    for i in range(days_range-1, -1, -1):
        d = filter_date - timedelta(days=i)
        count = db.session.query(func.count(Tracking.id)).filter(
            Tracking.personnel_id == personnel.id,
            func.date(Tracking.timestamp) == d
        ).scalar()
        chart_labels.append(d.strftime('%d-%m'))
        chart_data.append(count)

    context = {
        'personnel': personnel,
        'violation_summary': violation_summary,
        'filter_date_str': filter_date_str,
        'chart_labels': chart_labels,
        'chart_data': chart_data,
        'name': personnel.name
    }
    return render_template('employee_panel/dashboard.html', **context)

@bp.route('/presence_history')
@login_required
@employee_required
def presence_history():
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        flash("Your account is not linked to personnel data. Please contact admin.", "danger")
        logout_user()
        return redirect(url_for('auth.login'))

    date_str = request.args.get('date')
    selected_date = None
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            flash("Invalid date format. Showing all history.", "warning")
            selected_date = None

    presence_data = _get_employee_presence_data(target_date=selected_date, personnel_id=personnel.id)

    context = {
        'presence_data': presence_data,
        'selected_date': selected_date.isoformat() if selected_date else None,
        'personnel': personnel # Pass personnel for template display
    }
    return render_template('employee_panel/presence_history.html', **context)

@bp.route('/take_image')
@login_required
@employee_required
def take_image():
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        flash("Your account is not linked to personnel data.", "danger")
        return redirect(url_for('employee.dashboard'))
    
    return render_template('employee_panel/capture.html', name=personnel.name, personnel_id=personnel.id)