; Save EIS windata objects to .sav files.

pro export_windata, wd_files, eis_files, wl0

catch, error
if error ne 0 then exit, status=1

nfiles = n_elements(eis_files)
for i=0, nfiles-1 do begin
  wd_file = wd_files[i]
  eis_file = eis_files[i]
  wd = eis_getwindata(eis_file, wl0, /refill)
  if (wd.hdr.slit_id eq '40"') or (wd.hdr.slit_id eq '266"') then begin
    print, 'Reading slot.'
    eis_readslot, eis_file, wd, win=wl0, /estimate
  endif
  save, wd, filename=wd_file
endfor

end
