; Apply eis_prep to the files received from CLI, and ingest them into the EIS
; directory structure.
; WARNING: existing files will not be overwritten.
;
; Usage:
;   prep_one, "/path/to/fits"
;   prep, ["/path/to/fits_1", "/path/to/fits_2", "/path/to/fits_3" ...]

pro prep_one, fits_path

  path_split = strsplit(fits_path, '/', /extract)
  basename = path_split[-1]

  eis_prep, fits_path, /default, /quiet, /save, /retain, /correct_sensitivity

  file_l1 = mg_streplace(basename, 'eis_l0_(.*)', 'eis_l1_$1')
  file_er = mg_streplace(basename, 'eis_l0_(.*)', 'eis_er_$1')
  eis_ingest, [file_l1, file_er]
  ; clean files in case eis_ingest failed
  file_delete, file_l1, file_er, /allow_nonexistent

end


pro prep, files

  catch, error
  if error ne 0 then exit, status=1

  n_files = n_elements(files)

  for i=0, n_files-1 do begin
    fits_path = files[i]
    print, "Preparing ", fits_path
    prep_one, fits_path
  endfor

end
