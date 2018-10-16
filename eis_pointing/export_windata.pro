; Save EIS windata objects to .sav files.

pro export_windata, wd_files, l0_files, wl0

catch, error
if error ne 0 then exit, status=1

nfiles = n_elements(l0_files)
for i=0, nfiles-1 do begin
  wd_file = wd_files[i]
  l0_file = l0_files[i]
  wd = eis_getwindata(l0_file, wl0, /refill)
  if (wd.hdr.slit_id eq '40"') or (wd.hdr.slit_id eq '266"') then begin
    print, 'Reading slot.'
    eis_readslot, l0_file, wd, win=wl0, /estimate
  endif
  save, wd, filename=wd_file
endfor

end
