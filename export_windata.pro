; Save EIS windata objects to .sav files.

pro export_windata, wd_files, l0_files, aia_band

nfiles = n_elements(l0_files)
for i=0, nfiles-1 do begin
  wd_file = wd_files[i]
  l0_file = l0_files[i]
  ; TODO: join all wavelengths needed to compute the AIA
  ; emission in `eis_aia_emission.py`
  wd = eis_getwindata(l0_file, 195.119, /refill)
  save, wd, filename=wd_file
endfor

end
