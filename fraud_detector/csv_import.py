import csv
import io
from datetime import datetime
import logging
import time
import sys
import os
from django.db import transaction
from django.utils.dateparse import parse_datetime
from django.utils import timezone
import re
from .models import Transaction


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.getcwd(), 'csv_import.log'))
    ]
)
logger = logging.getLogger(__name__)

class CSVImportError(Exception):
    pass

def parse_boolean(value):
    if value is None:
        return False
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    if isinstance(value, str):
        if not value.strip():
            return False
        return value.lower() in ('true', 'yes', 'y', 't', '1')
    
    return False

def parse_timestamp(ts_string):
    if not ts_string or not isinstance(ts_string, str) or not ts_string.strip():
        return timezone.now()
    
    ts_string = ts_string.strip()
    
    try:
        pattern = r'(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\s+(AM|PM)'
        match = re.match(pattern, ts_string)
        
        if match:
            date_part, hour, minute, second, meridian = match.groups()
            
            month, day, year = map(int, date_part.split('/'))
            
            hour = int(hour)
            if meridian.upper() == 'PM' and hour < 12:
                hour += 12
            elif meridian.upper() == 'AM' and hour == 12:
                hour = 0
            
            naive_dt = datetime(year, month, day, hour, int(minute), int(second))
            return timezone.make_aware(naive_dt)
    except Exception:
        pass
    
    formats = [
        '%Y/%m/%d %H:%M',
    ]
    
    for format in formats:
        try:
            naive_dt = datetime.strptime(ts_string, format)
            return timezone.make_aware(naive_dt)
        except ValueError:
            continue
    
    try:
        parsed = parse_datetime(ts_string)
        if parsed:
            if timezone.is_naive(parsed):
                return timezone.make_aware(parsed)
            return parsed
    except Exception:
        pass
    
    return timezone.now()


def map_header_to_field(header_text, position=None, headers=None):


    processed_header = header_text.strip().lower().replace(' ', '')
    
    mappings = {
        'timestamp': 'timestamp',
        'frombank': 'from_bank',
        'tobank': 'to_bank',
        'amountreceived': 'amount_received',
        'receivingcurrency': 'receiving_currency',
        'amountpaid': 'amount_paid',
        'paymentcurrency': 'payment_currency',
        'paymentformat': 'payment_format',
        'islaundering': 'is_laundering'
    }
    
    if processed_header in mappings:
        return mappings[processed_header]
    
    if processed_header == 'account' and position is not None and headers is not None:
        if position > 0:
            prev_header = headers[position-1].strip().lower().replace(' ', '')
            if 'frombank' in prev_header:
                return 'from_account'
            elif 'tobank' in prev_header:
                return 'to_account'
    
    return None

def read_csv_file(file_obj, delimiter=',', encoding='utf-8', has_header=True):
    start_time = time.time()
    
    try:
        file_obj.seek(0, os.SEEK_END)
        file_obj.seek(0)
        
        file_content = file_obj.read().decode(encoding)
        
        
        delimiters = [',', ';', '\t', '|']
        if delimiter not in delimiters:
            delimiters.append(delimiter)
        
        
        logger.info(f"Parsing CSV data with delimiter '{delimiter}'...")
        parse_start = time.time()
        
        if delimiter == ' ':
            lines = file_content.splitlines()
            
            if has_header:
                header_line = lines[0]
                
                header_fields = []
                current_field = ""
                for word in header_line.split():
                    current_field += word + " "
                    
                    clean_field = current_field.strip().lower().replace(' ', '')
                    if clean_field in ['timestamp', 'frombank', 'fromaccount', 'tobank', 'toaccount', 
                                     'amountreceived', 'receivingcurrency', 'amountpaid', 
                                     'paymentcurrency', 'paymentformat', 'islaundering']:
                        header_fields.append(current_field.strip())
                        current_field = ""
                
                if not header_fields or len(header_fields) < 5:
                    header_fields = header_line.split()
                
                logger.info(f"Identified header fields: {header_fields}")
                
                header_mapping = {}
                for i, header in enumerate(header_fields):
                    mapped_field = map_header_to_field(header)
                    if mapped_field:
                        header_mapping[mapped_field] = i
                        logger.info(f"Mapped '{header}' to '{mapped_field}'")
                
                rows = []
                for i in range(1, len(lines)):
                    line = lines[i]
                    if not line.strip():
                        continue
                    
                    fields = line.split()
                    if len(fields) < len(header_fields):
                        continue
                    
                    row_dict = {}
                    expected_fields = ['timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
                                    'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
                                    'payment_format', 'is_laundering']
                    
                    for field in expected_fields:
                        if field in header_mapping and header_mapping[field] < len(fields):
                            row_dict[field] = fields[header_mapping[field]]
                        else:
                            row_dict[field] = ''
                    
                    rows.append(row_dict)
                    
                    if i <= 3:
                        logger.info(f"Sample row {i}: {row_dict}")
            else:
                rows = []
                expected_fields = ['timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
                                'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
                                'payment_format', 'is_laundering']
                
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    
                    fields = line.split()
                    row_dict = {}
                    
                    for j, field in enumerate(expected_fields):
                        if j < len(fields):
                            row_dict[field] = fields[j]
                        else:
                            row_dict[field] = ''
                    
                    rows.append(row_dict)
                    
                    if i < 3:
                        logger.info(f"Sample row {i+1}: {row_dict}")
        else:
            csv_reader = csv.reader(io.StringIO(file_content), delimiter=delimiter)
            
            expected_fields = [
                'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
                'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
                'payment_format', 'is_laundering'
            ]
            
            if has_header:
                file_headers = next(csv_reader)
                logger.info(f"File headers: {file_headers}")
                
                header_mapping = {}
                for i, header in enumerate(file_headers):
                    mapped_field = map_header_to_field(header, i, file_headers)
                    if mapped_field:
                        header_mapping[mapped_field] = i
                        logger.info(f"Mapped '{header}' to '{mapped_field}'")

                
                
                missing_fields = [f for f in expected_fields if f not in header_mapping]
                if missing_fields:
                    logger.warning(f"The following fields missing: {missing_fields}")
            else:
                header_mapping = {field: i for i, field in enumerate(expected_fields) if i < len(file_content.splitlines()[0].split(delimiter))}
                logger.info("No headers - mapping fields by position")
            
            rows = []
            row_count = 0
            
            for row in csv_reader:
                if not row or all(not cell for cell in row):
                    continue
                    
                row_dict = {}
                for field in expected_fields:
                    if field in header_mapping and header_mapping[field] < len(row):
                        row_dict[field] = row[header_mapping[field]]
                    else:
                        row_dict[field] = ''
                
                rows.append(row_dict)
                
                row_count += 1
                if row_count % 10000 == 0:
                    logger.info(f"Processed {row_count} rows so far")
                    
                if row_count <= 3:
                    logger.info(f"Sample row {row_count}: {row_dict}")
        
        logger.info(f"CSV parsing completed in {time.time() - parse_start:.2f} seconds")
        logger.info(f"Total rows read: {len(rows)}")
        logger.info(f"Read CSV file completed in {time.time() - start_time:.2f} seconds")
        
        for row in rows[:1]:
            logger.info(f"First row data: {row}")
        
        return rows
    
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}", exc_info=True)
        raise CSVImportError(f"Failed to read CSV file: {str(e)}")

def process_and_import_csv(file_obj, delimiter=',', encoding='utf-8', has_header=True):
    overall_start_time = time.time()
    logger.info("=== Starting CSV import ===")
    
    success_count = 0
    error_count = 0
    error_messages = []
    batch_size = 1000
    try:
        logger.info("Reading CSV")
        read_start = time.time()
        rows = read_csv_file(file_obj, delimiter, encoding, has_header)
        logger.info(f"Read {len(rows)} rows in {time.time() - read_start:.2f} seconds")
        
        logger.info("Importing started")
        total_rows = len(rows)
        batches = []
        current_batch = []
        
        for ids, row_data in enumerate(rows):
            try:
                if ids < 3:
                    logger.info(f"Processing row {ids+1}: {row_data}")
                
                timestamp_str = row_data.get('timestamp')
                timestamp = parse_timestamp(timestamp_str)
                
                try:
                    from_bank_str = row_data.get('from_bank', '')
                    if from_bank_str:
                        from_bank_str = ''.join(c for c in from_bank_str if c.isdigit() or c == '.')
                        from_bank = int(float(from_bank_str))
                    else:
                        from_bank = 0
                except (ValueError, TypeError):
                    from_bank = 0
                    
                try:
                    to_bank_str = row_data.get('to_bank', '')
                    if to_bank_str:
                        to_bank_str = ''.join(c for c in to_bank_str if c.isdigit() or c == '.')
                        to_bank = int(float(to_bank_str))
                    else:
                        to_bank = 0
                except (ValueError, TypeError):
                    to_bank = 0
                    
                try:
                    amount_received_str = row_data.get('amount_received', '')
                    if amount_received_str:
                        amount_received_str = ''.join(c for c in amount_received_str if c.isdigit() or c in '.-')
                        amount_received = float(amount_received_str)
                    else:
                        amount_received = 0.0
                except (ValueError, TypeError):
                    amount_received = 0.0
                    
                try:
                    amount_paid_str = row_data.get('amount_paid', '')
                    if amount_paid_str:
                        amount_paid_str = ''.join(c for c in amount_paid_str if c.isdigit() or c in '.-')
                        amount_paid = float(amount_paid_str)
                    else:
                        amount_paid = 0.0
                except (ValueError, TypeError):
                    amount_paid = 0.0
                
                is_laundering = parse_boolean(row_data.get('is_laundering', False))
                
                from_account = str(row_data.get('from_account', '') or '')
                to_account = str(row_data.get('to_account', '') or '')
                receiving_currency = str(row_data.get('receiving_currency', '') or 'USD')
                payment_currency = str(row_data.get('payment_currency', '') or 'USD')
                payment_format = str(row_data.get('payment_format', '') or 'Unknown')
                
                
                if timestamp is None:
                    timestamp = timezone.now()
                
                transaction_data = {
                    'timestamp': timestamp,
                    'from_bank': from_bank,
                    'from_account': from_account,
                    'to_bank': to_bank,
                    'to_account': to_account,
                    'amount_received': amount_received,
                    'receiving_currency': receiving_currency,
                    'amount_paid': amount_paid,
                    'payment_currency': payment_currency,
                    'payment_format': payment_format,
                    'is_laundering': is_laundering,
                }
                
                if ids < 3:
                    logger.info(f"Processed transaction data for row {ids+1}: {transaction_data}")
                
                current_batch.append(transaction_data)
                
                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    current_batch = []
                
                if (ids + 1) % 10000 == 0 or ids + 1 == total_rows:
                    progress = (ids + 1) / total_rows * 100
                    elapsed = time.time() - overall_start_time
                    est_total = elapsed / ((ids + 1) / total_rows)
                    remaining = est_total - elapsed
                    logger.info(f"Processed {ids + 1}/{total_rows} rows ({progress:.1f}%) - Est. remaining: {remaining/60:.1f} minutes")
                
            except Exception as e:
                error_count += 1
                error_message = f"Error in row {ids+1}: {str(e)}"
                error_messages.append(error_message)
                logger.error(error_message)
                logger.error(f"Problem row data: {row_data}")
        
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Importing {len(batches)} batches with up to {batch_size} transactions each")
        
        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()
            with transaction.atomic():
                for trans_data in batch:
                    Transaction.objects.create(**trans_data)
                
                success_count += len(batch)
                
            batch_time = time.time() - batch_start
            logger.info(f"Batch {batch_idx+1}/{len(batches)} imported successfully in {batch_time:.2f} seconds ({len(batch)/batch_time:.1f} rows/sec)")
        
        total_time = time.time() - overall_start_time
        logger.info(f"=== CSV import completed in {total_time:.2f} seconds ===")
        logger.info(f"Success: {success_count} transactions, Errors: {error_count}")
        
        return success_count, error_count, error_messages
    
    except CSVImportError as e:
        logger.error(f"CSV Import Error: {str(e)}")
        return 0, 1, [str(e)]
    except Exception as e:
        logger.error(f"Unexpected error during CSV import: {str(e)}", exc_info=True)
        return 0, 1, [f"Unexpected error: {str(e)}"]